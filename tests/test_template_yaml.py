"""Tests for YAML-based template loading."""

import pytest
from prompti.engine import PromptEngine, Setting
from prompti.template import PromptTemplate
from prompti.message import Message


@pytest.mark.asyncio
async def test_format_yaml():
    settings = Setting(template_paths=["./prompts"])
    engine = PromptEngine.from_setting(settings)
    messages = await engine.format("summary", {"summary": "Hello"})
    assert messages[-1].content == "Hello"


class TestPromptTemplateFormat:
    """Test suite for PromptTemplate.format method."""

    def test_format_simple_text(self):
        """Test formatting a simple text template."""
        template = PromptTemplate(
            id="test",
            name="test",
            version="1.0",
            required_variables=["name"],
            messages=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "text": "Hello {{ name }}!"
                        }
                    ]
                }
            ]
        )
        
        messages = template.format({"name": "World"})
        
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert messages[0].kind == "text"
        assert messages[0].content == "Hello World!"

    def test_format_multiple_messages(self):
        """Test formatting a template with multiple messages."""
        template = PromptTemplate(
            id="test",
            name="test",
            version="1.0",
            required_variables=["topic", "details"],
            messages=[
                {
                    "role": "system",
                    "parts": [
                        {
                            "type": "text",
                            "text": "You are an expert on {{ topic }}."
                        }
                    ]
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "text": "Please explain {{ details }}."
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "text",
                            "text": "I'll help you understand {{ topic }} and {{ details }}."
                        }
                    ]
                }
            ]
        )
        
        messages = template.format({"topic": "AI", "details": "neural networks"})
        
        assert len(messages) == 3
        assert messages[0].role == "system"
        assert messages[0].content == "You are an expert on AI."
        assert messages[1].role == "user"
        assert messages[1].content == "Please explain neural networks."
        assert messages[2].role == "assistant"
        assert messages[2].content == "I'll help you understand AI and neural networks."

    def test_format_with_file_parts(self):
        """Test formatting a template with file parts."""
        template = PromptTemplate(
            id="test",
            name="test",
            version="1.0",
            messages=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "text": "Please analyze this file:"
                        },
                        {
                            "type": "file",
                            "file": "/path/to/document.pdf"
                        }
                    ]
                }
            ]
        )
        
        messages = template.format({})
        
        assert len(messages) == 2
        assert messages[0].role == "user"
        assert messages[0].kind == "text"
        assert messages[0].content == "Please analyze this file:"
        assert messages[1].role == "user"
        assert messages[1].kind == "file"
        assert messages[1].content == "/path/to/document.pdf"

    def test_format_jinja2_for_loop(self):
        """Test formatting a template with Jinja2 for loop."""
        template = PromptTemplate(
            id="test",
            name="test",
            version="1.0",
            required_variables=["items"],
            messages=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "text": "Here are the items:\n{% for item in items %}{{ loop.index }}. {{ item }}\n{% endfor %}"
                        }
                    ]
                }
            ]
        )
        
        messages = template.format({"items": ["apple", "banana", "cherry"]})
        
        assert len(messages) == 1
        assert messages[0].content == "Here are the items:\n1. apple\n2. banana\n3. cherry\n"

    def test_format_jinja2_if_else(self):
        """Test formatting a template with Jinja2 if/else statements."""
        template = PromptTemplate(
            id="test",
            name="test",
            version="1.0",
            required_variables=["user_type", "name"],
            messages=[
                {
                    "role": "assistant",
                    "parts": [
                        {
                            "type": "text",
                            "text": "{% if user_type == 'premium' %}Welcome back, premium member {{ name }}! You have access to all features.{% else %}Hello {{ name }}! Consider upgrading to premium for more features.{% endif %}"
                        }
                    ]
                }
            ]
        )
        
        # Test premium user
        messages = template.format({"user_type": "premium", "name": "Alice"})
        assert messages[0].content == "Welcome back, premium member Alice! You have access to all features."
        
        # Test regular user
        messages = template.format({"user_type": "regular", "name": "Bob"})
        assert messages[0].content == "Hello Bob! Consider upgrading to premium for more features."

    def test_format_complex_jinja2_logic(self):
        """Test formatting a template with complex Jinja2 logic combining loops and conditionals."""
        template = PromptTemplate(
            id="test",
            name="test",
            version="1.0",
            required_variables=["tasks", "priority_threshold"],
            messages=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "text": """Task Report:
{% for task in tasks -%}
{% if task.priority >= priority_threshold -%}
🔥 HIGH: {{ task.name }} (Priority: {{ task.priority }})
{% else -%}
📝 NORMAL: {{ task.name }} (Priority: {{ task.priority }})
{% endif -%}
{% endfor -%}

{% set high_priority_count = tasks | selectattr('priority', '>=', priority_threshold) | list | length -%}
Total high-priority tasks: {{ high_priority_count }}"""
                        }
                    ]
                }
            ]
        )
        
        tasks_data = {
            "tasks": [
                {"name": "Fix bug", "priority": 9},
                {"name": "Update docs", "priority": 3},
                {"name": "Security patch", "priority": 10},
                {"name": "Refactor code", "priority": 5}
            ],
            "priority_threshold": 8
        }
        
        messages = template.format(tasks_data)
        content = messages[0].content
        
        assert "🔥 HIGH: Fix bug (Priority: 9)" in content
        assert "📝 NORMAL: Update docs (Priority: 3)" in content
        assert "🔥 HIGH: Security patch (Priority: 10)" in content
        assert "📝 NORMAL: Refactor code (Priority: 5)" in content
        assert "Total high-priority tasks: 2" in content

    def test_format_missing_required_variables(self):
        """Test that missing required variables raise KeyError."""
        template = PromptTemplate(
            id="test",
            name="test",
            version="1.0",
            required_variables=["name", "age"],
            messages=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "text": "Hello {{ name }}, you are {{ age }} years old."
                        }
                    ]
                }
            ]
        )
        
        with pytest.raises(KeyError, match="missing variables: \\['age'\\]"):
            template.format({"name": "Alice"})

    def test_format_empty_template(self):
        """Test formatting an empty template."""
        template = PromptTemplate(
            id="test",
            name="test",
            version="1.0",
            messages=[]
        )
        
        messages = template.format({})
        assert len(messages) == 0

    def test_format_with_tag_parameter(self):
        """Test that tag parameter is accepted (though not currently used)."""
        template = PromptTemplate(
            id="test",
            name="test",
            version="1.0",
            messages=[
                {
                    "role": "user",
                    "parts": [
                        {
                            "type": "text",
                            "text": "Hello world"
                        }
                    ]
                }
            ]
        )
        
        # Should not raise an error
        messages = template.format({}, tag="test-tag")
        assert len(messages) == 1
        assert messages[0].content == "Hello world"

    def test_format_multiline_text(self):
        """Test formatting a template with multiline text."""
        template = PromptTemplate(
            id="test",
            name="test",
            version="1.0",
            required_variables=["project_name", "features"],
            messages=[
                {
                    "role": "system",
                    "parts": [
                        {
                            "type": "text",
                            "text": """You are a technical writer for {{ project_name }}.

Please create documentation that covers:
{% for feature in features -%}
- {{ feature }}
{% endfor %}

Make it clear and comprehensive."""
                        }
                    ]
                }
            ]
        )
        
        messages = template.format({
            "project_name": "MyApp",
            "features": ["authentication", "data processing", "API endpoints"]
        })
        
        expected_content = """You are a technical writer for MyApp.

Please create documentation that covers:
- authentication
- data processing
- API endpoints


Make it clear and comprehensive."""
        
        assert messages[0].content == expected_content
