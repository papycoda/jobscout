"""Unit tests for backend LLM helpers with latest OpenAI defaults."""

import importlib
import sys
import types
from types import SimpleNamespace
from typing import Optional

import pytest


def _make_fake_openai(monkeypatch, recorder):
    """Create a fake openai module that records calls and returns stubbed responses."""

    class FakeResponse:
        def __init__(self, content: str):
            self.choices = [
                SimpleNamespace(
                    message=SimpleNamespace(content=content),
                    finish_reason="stop",
                )
            ]
            self.usage = {"prompt_tokens": 1, "completion_tokens": 1}

    class FakeCompletions:
        def create(self, model, messages, max_tokens=None, max_completion_tokens=None, **kwargs):
            recorder["model"] = model
            recorder["messages"] = messages
            recorder["max_tokens"] = max_tokens
            recorder["max_completion_tokens"] = max_completion_tokens

            system_prompt = messages[0]["content"]
            if "job description analyzer" in system_prompt:
                return FakeResponse(
                    '{"must_haves":["python","sql"],"nice_to_haves":["docker"],'
                    '"seniority":"mid","remote_eligible":true,"constraints":[],'
                    '"years_experience":5}'
                )

            return FakeResponse(
                '{"skills":["python"],"seniority":"senior","role_focus":["backend"],'
                '"years_experience":10,"keywords":["engineer"]}'
            )

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self):
            self.chat = FakeChat()

    fake_module = types.ModuleType("openai")

    def _factory(api_key: str):
        recorder["api_key"] = api_key
        return FakeClient()

    fake_module.OpenAI = _factory
    monkeypatch.setitem(sys.modules, "openai", fake_module)


@pytest.fixture
def reload_llm(monkeypatch):
    """Reload backend.llm with patched environment and fake openai module."""

    def _reload(openai_model: Optional[str]):
        recorder = {}
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        if openai_model is None:
            monkeypatch.delenv("OPENAI_MODEL", raising=False)
        else:
            monkeypatch.setenv("OPENAI_MODEL", openai_model)

        _make_fake_openai(monkeypatch, recorder)

        if "backend.llm" in sys.modules:
            del sys.modules["backend.llm"]
        import backend.llm as llm

        importlib.reload(llm)
        return llm, recorder

    return _reload


def test_default_model_is_gpt5_mini(monkeypatch, reload_llm):
    llm, recorder = reload_llm(openai_model=None)

    profile = llm.extract_profile("Python engineer with 10 years experience")

    assert profile["skills"] == ["python"]
    assert recorder["model"] == "gpt-5-mini"
    assert recorder["max_completion_tokens"] == 2000


def test_invalid_model_falls_back_to_gpt5(monkeypatch, reload_llm):
    llm, recorder = reload_llm(openai_model="not-a-real-model")

    requirements = llm.extract_job_requirements("We need Python and SQL skills")

    assert set(requirements["must_haves"]) == {"python", "sql"}
    assert recorder["model"] == "gpt-5-mini"
    assert recorder["max_completion_tokens"] == 2000
