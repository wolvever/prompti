"""
Trace service for reporting model calls to external services.
"""
from typing import Any, Dict, List, Optional, Union
import asyncio
import logging
import time
from dataclasses import dataclass, field

import httpx
from opentelemetry import trace


# Setup logger
logger = logging.getLogger(__name__)
# Get tracer for this module
_tracer = trace.get_tracer(__name__)


@dataclass
class TraceEvent:
    """Event data structure for model call trace."""
    
    # Request information
    template_name: str
    template_id: Optional[str] = ""
    template_version: Optional[str] = None
    variant: Optional[str] = None

    model: str = ""
    messages_template: List[Dict[str, Any]] = field(default_factory=list)
    variables: Optional[Dict[str, Any]] = None

    # Request information
    llm_request_body: Dict[str, Any] = field(default_factory=dict)
    llm_response_body: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    request_id: str = ""
    timestamp: float = field(default_factory=time.time)
    conversation_id: str = ""
    user_id: str = ""
    duration_ms: Optional[float] = None
    token_usage: Dict[str, int] = field(default_factory=dict)
    error: Optional[str] = None
    source: Optional[str] = None
    span_id: Optional[str] = None
    parent_span_id: Optional[str] = None
    perf_metrics: Optional[Dict[str, Any]] = None
    
    # Additional context
    ext: Dict[str, Any] = field(default_factory=dict)


class TraceService:
    """
    Service for reporting model call trace to external endpoints.
    """
    
    def __init__(
        self, 
        endpoint_url: str,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        max_retries: int = 3,
        enabled: bool = True
    ):
        """
        Initialize the trace service.
        
        Args:
            endpoint_url: URL of the trace endpoint
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts for failed reports
            enabled: Whether trace reporting is enabled
        """
        self.endpoint_url = endpoint_url
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.enabled = enabled
        self._http_client = None
        self._sync_http_client = None
        
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._http_client is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                
            self._http_client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=headers
            )
        return self._http_client
    
    def _get_sync_client(self) -> httpx.Client:
        """Get or create the synchronous HTTP client."""
        if self._sync_http_client is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
                
            self._sync_http_client = httpx.Client(
                timeout=self.timeout,
                headers=headers
            )
        return self._sync_http_client

    async def aclose(self):
        """Close the HTTP client if it exists."""
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
    
    def close(self):
        """Close the synchronous HTTP client if it exists."""
        if self._sync_http_client is not None:
            self._sync_http_client.close()
            self._sync_http_client = None
    
    async def areport(self, event: TraceEvent) -> bool:
        """
        Report a model call event to the trace endpoint.
        
        Args:
            event: The trace event to report
            
        Returns:
            True if the report was successfully sent, False otherwise
        """
        if not self.enabled:
            logger.debug("Trace reporting is disabled")
            return True
            
        with _tracer.start_as_current_span("trace.report"):
            client = await self._get_client()
            
            # Convert event to serializable dict
            payload = {
                "template_name": event.template_name,
                "template_id": event.template_id,
                "template_version": event.template_version,
                "variant": event.variant,
                "model": event.model,
                "messages_template": event.messages_template,
                "variables": event.variables,
                "llm_request_body": event.llm_request_body,
                "llm_response_body": event.llm_response_body,
                "request_id": event.request_id,
                "user_id": event.user_id,
                "timestamp": event.timestamp,
                "conversation_id": event.conversation_id,
                "token_usage": event.token_usage,
                "error": event.error,
                "source": event.source,
                "span_id": event.span_id,
                "parent_span_id": event.parent_span_id,
                "ext": event.ext,
                "perf_metrics": event.perf_metrics
            }
            url = self.endpoint_url + "/trace/llm-message/dump"

            # Try to send the report with retries
            for attempt in range(self.max_retries):
                try:
                    response = await client.post(
                        url,
                        json=payload,
                        timeout=self.timeout
                    )
                    
                    if response.status_code < 400:
                        logger.debug(f"Trace report sent successfully: {response.status_code}")
                        return True
                    
                    logger.warning(
                        f"Trace report failed (attempt {attempt+1}/{self.max_retries}): "
                        f"Status {response.status_code}, {response.text}"
                    )
                    
                    # Wait before retry (exponential backoff)
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt * 0.1)
                        
                except Exception as e:
                    logger.warning(
                        f"Trace report exception (attempt {attempt+1}/{self.max_retries}): {str(e)}"
                    )
                    
                    # Wait before retry (exponential backoff)
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(2 ** attempt * 0.1)
            
            # All attempts failed
            logger.error(f"Trace report failed after {self.max_retries} attempts")
            return False
    
    def report(self, event: TraceEvent) -> bool:
        """
        Synchronous version: Report a model call event to the trace endpoint.
        
        Args:
            event: The trace event to report
            
        Returns:
            True if the report was successfully sent, False otherwise
        """
        if not self.enabled:
            logger.debug("Trace reporting is disabled")
            return True
            
        client = self._get_sync_client()
        
        # Convert event to serializable dict
        payload = {
            "template_name": event.template_name,
            "template_id": event.template_id,
            "template_version": event.template_version,
            "variant": event.variant,
            "model": event.model,
            "messages_template": event.messages_template,
            "variables": event.variables,
            "llm_request_body": event.llm_request_body,
            "llm_response_body": event.llm_response_body,
            "request_id": event.request_id,
            "user_id": event.user_id,
            "timestamp": event.timestamp,
            "conversation_id": event.conversation_id,
            "token_usage": event.token_usage,
            "error": event.error,
            "source": event.source,
            "span_id": event.span_id,
            "parent_span_id": event.parent_span_id,
            "ext": event.ext,
            "perf_metrics": event.perf_metrics
        }
        url = self.endpoint_url + "/trace/llm-message/dump"

        # Try to send the report with retries
        for attempt in range(self.max_retries):
            try:
                response = client.post(
                    url,
                    json=payload,
                    timeout=self.timeout
                )
                
                if response.status_code < 400:
                    logger.debug(f"Trace report sent successfully: {response.status_code}")
                    return True
                
                logger.warning(
                    f"Trace report failed (attempt {attempt+1}/{self.max_retries}): "
                    f"Status {response.status_code}, {response.text}"
                )
                
                # Wait before retry (exponential backoff)
                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(2 ** attempt * 0.1)
                    
            except Exception as e:
                logger.warning(
                    f"Trace report exception (attempt {attempt+1}/{self.max_retries}): {str(e)}"
                )
                
                # Wait before retry (exponential backoff)
                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(2 ** attempt * 0.1)
        
        # All attempts failed
        logger.error(f"Trace report failed after {self.max_retries} attempts")
        return False
