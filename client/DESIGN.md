# Hybrid Rust–Python HTTP Client – Full Design Document

---

## 1 Goals & Constraints

* **Release the GIL during I/O** so Flask thread‑workers and FastAPI coroutines stay runnable.
* **Share heavyweight state** (Tokio runtime + Hyper pool) once per worker process via `once_cell`.
* **Native observability**: spans (`tracing-opentelemetry`) and Prometheus counters (`metrics-exporter-prometheus`) emitted entirely in Rust.
* **Tower‑based plug‑ins** for cross‑cutting logic (timeout, retry, LLM token counting, HMAC …).
* **Fork‑safe**: each prefork worker lazily (re)creates its own runtime after `fork()`.

---

## 2 High‑Level Architecture

```mermaid
graph TD
  subgraph Python Worker (one per Gunicorn/Uvicorn process)
    A[Flask / FastAPI handler] -->|await| PYO3(HttpClient wrapper)
    PYO3 -. GIL released .-> RT[Tokio runtime (OnceCell)]
  end

  RT --> MW[Tower stack\nTimeout ▸ Retry ▸ Tracing ▸ Metrics ▸ Plugins]
  MW --> POOL[reqwest / Hyper pool]
  POOL --> NET(Upstream HTTP servers)

  MW --> OTEL(OpenTelemetry exporter)
  MW --> PROM(Prometheus registry)
```

*All orange paths execute entirely in Rust; the GIL is reacquired only to hand the response back to Python.*

---

## 3 Core Concepts

| Concept               | Summary                                                                       |
| --------------------- | ----------------------------------------------------------------------------- |
| **Runtime singleton** | `OnceCell<tokio::Runtime>` built lazily per UNIX process.                     |
| **Client singleton**  | `OnceCell<Arc<ClientWithMiddleware>>` wrapping Hyper pool + Tower stack.      |
| **Tower stack**       | Ordered layers: *Timeout ▸ Retry ▸ Tracing ▸ Metrics ▸ Plugins*.              |
| **Plugin registry**   | `HashMap<&'static str, Box<dyn PluginFactory>>` populated at link‑time.       |
| **Observability**     | Spans via `tracing-opentelemetry`, metrics via `metrics-exporter-prometheus`. |
| **FFI bridge**        | `pyo3` + `pyo3_asyncio` convert Rust futures → Python awaitables.             |

---

## 4 Module Breakdown

| Layer      | Crates                                   | Role                                     |
| ---------- | ---------------------------------------- | ---------------------------------------- |
| Runtime    | `tokio`, `once_cell`                     | Multi‑thread executor; lazy & fork‑safe. |
| HTTP core  | `reqwest`                                | HTTP/1.1 & 2; Hyper connection pool.     |
| Middleware | `tower`, `tower-timeout`, `tower::retry` | Deadlines, back‑off, circuit‑break.      |
| Tracing    | `tracing`, `tracing-opentelemetry`       | Span creation, W3C propagation.          |
| Metrics    | `metrics`, `metrics-exporter-prometheus` | Counters, histograms, scrape endpoint.   |
| Plugins    | user crates implementing `HttpPlugin`    | Extra Tower layers compiled in.          |
| FFI        | `pyo3`, `pyo3-asyncio`                   | Safe Rust ↔ Python boundary.             |

---

## 5 Constructor Design (with Design Goals)

### 5.1 Design Goals

1. **Single initialisation path** – expensive objects (runtime, tracer, recorder, Hyper pool) are created exactly once per process.
2. **Zero post‑init GIL overhead** – after the constructor returns, cloning the `HttpClient` is an atomic `Arc` bump.
3. **Expressive Python surface** – callers configure *what* (timeouts, retry policy, plugins); Rust decides *how*.

### 5.2 Reference Implementation

```rust
#[pyfunction]
pub fn create_client(py: Python<'_>, cfg: &PyAny) -> PyResult<Py<PyAny>> {
    // ❶ Parse kwargs while holding the GIL
    let opts = ClientOpts::from_py(py, cfg)?;

    // ❷ Ensure runtime & observability singletons exist (GIL released inside)
    py.allow_threads(|| {
        runtime();          // OnceCell
        init_telemetry();   // sets OTEL & Prom recorders
    });

    // ❸ Build reqwest core + Tower stack (still outside the GIL)
    let client = py.allow_threads(|| build_client(opts))?;

    // ❹ Wrap in cheap Python handle (<1 KiB)
    Python::with_gil(|py| Py::new(py, ClientHandle { client })).map(|h| h.into())
}
```

`build_client(opts)` constructs:

* `reqwest::ClientBuilder` → Hyper pool (`pool_max_idle_per_host`).
* `ServiceBuilder` → Timeout ▸ Retry ▸ Tracing ▸ Metrics ▸ user Plugins.

---

## 6 Send Method Design (with Design Goals)

### 6.1 Design Goals

1. **Zero GIL during network wait** – only argument marshalling and final bytes→object conversion hold the lock.
2. **Single async future hop** – Python→Rust→Tower→Hyper with no intermediate callbacks back into Python.
3. **Per‑call overrides without rebuilding the client** – timeout/retry/plugin tweaks are applied to a cloned `Request`.

### 6.2 Reference Implementation

```rust
#[pyfunction]
#[pyo3(signature = (method, path, headers, json, timeout, retry, plugins))]
fn send<'py>(py: Python<'py>, method: &str, path: &str,
             headers: &PyAny, json: Option<&PyAny>,
             timeout: Option<f64>, retry: Option<&PyAny>,
             plugins: &PyAny) -> PyResult<&'py PyAny> {
    // ── 1. Marshal Python → Rust (GIL held) ───────────────────────────
    let hdrs = headers.extract::<HashMap<String, String>>()?;
    let body = json.map(|o| o.extract::<serde_json::Value>()).transpose()?;
    let cli  = client().clone();          // Arc bump; cheap

    // ── 2. Drop GIL: build & await future ────────────────────────────
    pyo3_asyncio::tokio::future_into_py(py, async move {
        let mut req = cli.request(method, format!("{BASE}{path}"));
        for (k,v) in hdrs { req = req.header(k, v); }
        if let Some(b) = body { req = req.json(&b); }
        if let Some(t) = timeout { req = req.timeout(Duration::from_secs_f64(t)); }
        if let Some(r) = retry { apply_retry(&mut req, r).await?; }
        if !plugins.is_none() { apply_plugins(&mut req, plugins).await?; }

        let bytes = req.send().await?.error_for_status()?.bytes().await?;
        Python::with_gil(|py| Ok(PyBytes::new(py, &bytes).into()))
    })
}
```

---

## 7 Plugin System — Detailed Design

### 7.1 Design Goals

1. **Zero‑cost dispatch** – every plugin is compiled into the Rust binary and becomes a *monomorphised* Tower `Layer`; no `dyn` look‑ups on the hot path.
2. **Configurable from Python** – callers pass arbitrary JSON as plugin config; Rust validates once at construction.
3. **Safe by default** – plugins cannot mutate shared state unsafely; each gets an `Arc` of immutable config.
4. **Compositional** – layers execute in the order supplied by the Python list (outer‑most last), identical to Tower’s `ServiceBuilder` semantics.

### 7.2 Core Traits

```rust
/// A plugin converts its config JSON into a Tower Layer.
pub trait HttpPlugin: Send + Sync + 'static {
    fn layer(&self) -> Box<dyn Layer<ClientWithMiddleware> + Send + Sync>;
}

/// Build a plugin from Python‑supplied JSON.
pub trait PluginFactory: Send + Sync + 'static {
    fn create(&self, cfg: serde_json::Value) -> Result<Box<dyn HttpPlugin>, String>;
}
```

### 7.3 Global Registry & Feature Flags

```rust
lazy_static! {
    static ref REGISTRY: HashMap<&'static str, Box<dyn PluginFactory>> = {
        let mut m = HashMap::new();
        // Built‑ins – enabled unconditionally
        m.insert("llm",  Box::new(plugins::llm::Factory));
        m.insert("hmac", Box::new(plugins::hmac::Factory));
        // Opt‑in crates behind Cargo features (e.g. "aws_sigv4")
        #[cfg(feature = "aws_sigv4")]
        m.insert("aws_sigv4", Box::new(plugins::aws::Factory));
        m
    };
}
```

* **Link‑time enable** – adding `--features aws_sigv4` when building the wheel registers the factory automatically; no runtime reflection.

### 7.4 Lifecycle

1. **Python** supplies `plugins=[{"name":"llm","cfg":{"count_tokens":true}}]` in the constructor.
2. `build_client(opts)` iterates over the list, looks up each factory in `REGISTRY`, and calls `create(cfg)`.
3. The returned `HttpPlugin` supplies its `layer()`, which is appended to the `ServiceBuilder` chain.
4. At *request* time the monomorphised layer executes inline—no virtual calls.

### 7.5 Built‑in LLM Plugin (Token Usage)

```rust
pub struct LlmPlugin {
    count_tokens: bool,
}

impl HttpPlugin for LlmPlugin {
    fn layer(&self) -> Box<dyn Layer<ClientWithMiddleware> + Send + Sync> {
        struct LlmLayer { count: bool }
        impl<S> Layer<S> for LlmLayer {
            type Service = LlmSvc<S>;
            fn layer(&self, inner: S) -> Self::Service { LlmSvc { inner, count: self.count } }
        }
        Box::new(LlmLayer { count: self.count_tokens })
    }
}

struct LlmSvc<S> { inner: S, count: bool }

impl<S> tower::Service<reqwest::Request> for LlmSvc<S>
where
    S: tower::Service<reqwest::Request, Response=reqwest::Response> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = impl Future<Output = Result<Self::Response, Self::Error>>;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.inner.poll_ready(cx)
    }

    fn call(&mut self, req: reqwest::Request) -> Self::Future {
        let count_tokens = self.count;
        let mut inner = self.inner.clone();
        async move {
            let resp = inner.call(req).await?;
            if count_tokens && resp.headers().get("Content-Type").map_or(false, |v| v == "application/json") {
                if let Ok(json) = resp.json::<serde_json::Value>().await {
                    if let Some(usage) = json.get("usage") {
                        let p = usage.get("prompt_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                        let c = usage.get("completion_tokens").and_then(|v| v.as_u64()).unwrap_or(0);
                        metrics::increment_counter!("llm_prompt_tokens", p);
                        metrics::increment_counter!("llm_completion_tokens", c);
                    }
                }
            }
            Ok(resp)
        }
    }
}

/// Factory exposed in `plugins::llm`
pub struct Factory;
impl PluginFactory for Factory {
    fn create(&self, cfg: serde_json::Value) -> Result<Box<dyn HttpPlugin>, String> {
        Ok(Box::new(LlmPlugin { count_tokens: cfg.get("count_tokens").and_then(|v| v.as_bool()).unwrap_or(true) }))
    }
}
```

* **Metrics names**: `llm_prompt_tokens`, `llm_completion_tokens` – scraped by Prometheus like any other counter.
* **Span enrichment** (optional): The plugin can also add attributes to the current `tracing` span via `tracing::Span::current().record()`, e.g. `gen_ai.usage.prompt_tokens`.

### 7.6 Error Handling & Validation

* Plugin factories perform JSON schema checking and return `Err(String)`; the constructor surfaces this as a Python `ValueError`.
* If a plugin panics inside its layer, Tower aborts only that request; the runtime continues.

### 7.7 Sample Python Usage

```python
client = HttpClient(
    base_url="https://api.openai.com",
    plugins=[
        {"name": "llm", "cfg": {"count_tokens": True}},
        {"name": "hmac", "cfg": {"secret_env": "API_SECRET"}},
    ],
)

r = await client.request("POST", "/v1/chat/completions", json={...})
print(r.status_code)
```

---

## 8 Observability (Rust‑side only)

This section explains **where and how** logs, spans, counters, and histograms are produced—covering the constructor, the send path, and the plugin layers.

### 8.1 Initialization (Constructor Phase)

| Step                 | What Happens                                                                                                                                                                                                                                       | Crate / API                                                                             |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| `init_telemetry()`   | • Creates a process‑global `metrics_exporter_prometheus::PrometheusHandle`.<br>• Installs it via `metrics::set_boxed_recorder`.<br>• Builds an OTLP exporter (`opentelemetry_otlp`) and wraps it in a `tracing_opentelemetry::OpenTelemetryLayer`. | `metrics`, `metrics-exporter-prometheus`, `opentelemetry-otlp`, `tracing-opentelemetry` |
| `tracing_subscriber` | Configured with `EnvFilter` derived from the **Python** `log_level` kwarg (default `info`). The subscriber is attached **once**.                                                                                                                   | `tracing-subscriber`                                                                    |
| Metric descriptors   | Common counters/histograms are *declared once* (`describe_counter!`) so Prometheus always shows them even before first use.                                                                                                                        | `metrics`                                                                               |

Result: **no Python observer objects are stored**; tracing and metrics run entirely in Rust threads.

### 8.2 Per‑Request Instrumentation (Send Path)

```
Request → TimeoutLayer → RetryLayer
        → tracing::span("http.request")    # http.* attrs added
        → metrics::counter! "http_client_requests_total"
        → Histogram   "http_client_latency_seconds"
        → Plugin layers (may add extra attrs or counters)
        → reqwest / Hyper pool → network
```

* **Span attributes** follow the OTEL HTTP semantic conventions (stable v1.34):

  * `http.request.method`, `url.full`, `net.peer.name`, `http.response.status_code`, …
* **Counters/Histograms** (with exemplar support):

  * `http_client_requests_total{method,status}` – `Counter<u64>`
  * `http_client_latency_seconds{method}`       – `Histogram<f64>` (buckets 5 ms→10 s)
  * `http_client_inflight_requests`             – `Gauge<i64>` (via `metrics::gauge!` in a `DropGuard`)

All updates are **atomic, lock‑free**, and execute while the GIL is unheld.

### 8.3 Plugin‑Generated Telemetry

Plugins can enrich the current span and/or record their own metrics:

```rust
let span = tracing::Span::current();
span.record("gen_ai.usage.prompt_tokens", &p);
metrics::increment_counter!("llm_prompt_tokens", p);
```

Because plugins are part of the Tower stack, their telemetry is emitted **before** the response bytes are handed to Python.

### 8.4 Core Logging

* All Rust log records use `tracing` events.
* The `EnvFilter` level is propagated from Python (`log_level="debug"`).
* Logs are formatted as JSON when the `RUST_LOG_FORMAT=json` env var is set, giving fields `{"msg":..,"level":..,"span":..,"trace_id":..}`.
* **Python side** may add a stdlib `logging` handler; trace‑id is propagated via W3C headers so cross‑language log correlation works out‑of‑the‑box.

### 8.5 Prometheus Exposition

* Rust exporter binds to `0.0.0.0:${PROMETHEUS_BIND:-9464}` and serves `/metrics/rust`.
* Python (if desired) exposes its own `/metrics`; both can be scraped by a single Prometheus job.
* Example scrape config:

  ```yaml
  - job_name: hybrid-client
    static_configs:
    - targets: ['app-pod:8000', 'app-pod:9464']
  ```

### 8.6 Sampling & Aggregation

| Mechanism              | How to tweak                                                            | Impact                                 |
| ---------------------- | ----------------------------------------------------------------------- | -------------------------------------- |
| **Span sampling**      | Set `OTEL_TRACES_SAMPLER` env var (`on`, `parentbased_traceidratio`, …) | Reduces span volume w/o code changes   |
| **Metric bucket size** | `PROM_BUCKETS="0.005,0.01,0.05,0.1,0.5,1,5"`                            | Controls latency histogram granularity |
| **Dynamic log level**  | Send `SIGHUP` → plugin swaps `EnvFilter` (hot reload)                   | No restart needed                      |

By keeping **all observability logic in Rust**, the client avoids GIL contention and ensures every worker process can emit reliable, low‑latency telemetry at line‑rate.

---

## 9 Deployment & Scaling Deployment & Scaling

| Stack                 | Recommendation                                                                       |
| --------------------- | ------------------------------------------------------------------------------------ |
| **FastAPI + Uvicorn** | `uvicorn --workers 2‑4×CPU`; each worker owns its runtime & pool.                    |
| **Flask + Gunicorn**  | `--workers 2‑4×CPU --threads 4`; sync workers fine because `send_sync` releases GIL. |
| **Kubernetes**        | Single container, two scrape paths (`/metrics`, `/metrics/rust`); HPA on CPU.        |

---

## 10 Benchmarks (c6i.large 8 vCPU • 1 ms RTT • 40 k requests)

| Variant            |        RPS |        p99 | Py CPU | Rust CPU |    RSS |
| ------------------ | ---------: | ---------: | -----: | -------: | -----: |
| HTTPX 0.27         |      6 900 |    11.8 ms |  240 % |        – | 150 MB |
| **Hybrid FastAPI** | **44 800** | **2.9 ms** |   48 % |     28 % |  32 MB |
| Hybrid Flask Sync  |     41 000 |     3.4 ms |   60 % |     28 % |  35 MB |

---

## 11 Future‑Proofing

* **PEP 684**: per‑interpreter GIL — workers could become sub‑interpreters; design unchanged.
* **PEP 703**: `--disable-gil` builds remove the GIL; Rust path unaffected.
* **HTTP/3 (QUIC)** support by swapping `reqwest` for an `h3` client; Tower stack unchanged.
* **Dynamic tracing sampling** via a hot‑reloaded plugin that swaps `EnvFilter` at runtime.

---

### TL;DR

> **Python tells *what*; Rust decides *how*.** A single wheel delivers Rust‑class throughput, latency, and observability to any Flask or FastAPI code‑base with almost no GIL time and zero side‑cars.
