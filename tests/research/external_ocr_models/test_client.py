"""Клиент OpenRouter: ретраи, разбор usage и ошибок — без сети."""

import pytest
import requests

from research.external_ocr_models.client import OpenRouterClient, OpenRouterError


class FakeResponse:
    def __init__(self, status, body=None, text=""):
        self.status_code = status
        self._body = body
        self.text = text

    def json(self):
        return self._body


def _ok_body(text="{}", cost=0.0021):
    return {
        "id": "gen-1",
        "provider": "DeepSeek",
        "model": "deepseek/deepseek-v4.1-flash",
        "choices": [{"message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 1500,
            "completion_tokens": 700,
            "cost": cost,
            "completion_tokens_details": {"reasoning_tokens": 12},
        },
    }


def test_parses_usage_and_cost():
    client = OpenRouterClient("key", post=lambda *a, **k: FakeResponse(200, _ok_body()), sleep=lambda s: None)
    response = client.chat({"model": "x"})
    assert response.text == "{}"
    assert response.cost_usd == pytest.approx(0.0021)
    assert (response.prompt_tokens, response.completion_tokens, response.reasoning_tokens) == (1500, 700, 12)
    assert response.provider == "DeepSeek" and response.finish_reason == "stop" and response.attempts == 1


def test_retries_on_429_then_succeeds():
    calls = []

    def post(*args, **kwargs):
        calls.append(1)
        return FakeResponse(429, text="slow down") if len(calls) < 3 else FakeResponse(200, _ok_body())

    slept = []
    client = OpenRouterClient("key", attempts=5, post=post, sleep=slept.append)
    response = client.chat({})
    assert response.attempts == 3 and len(slept) == 2


def test_gives_up_after_attempts():
    client = OpenRouterClient("key", attempts=2, post=lambda *a, **k: FakeResponse(503), sleep=lambda s: None)
    with pytest.raises(OpenRouterError) as error:
        client.chat({})
    assert error.value.status == 503


def test_400_is_not_retried():
    calls = []

    def post(*args, **kwargs):
        calls.append(1)
        return FakeResponse(400, text="bad response_format")

    client = OpenRouterClient("key", attempts=5, post=post, sleep=lambda s: None)
    with pytest.raises(OpenRouterError) as error:
        client.chat({})
    assert error.value.status == 400 and len(calls) == 1


def test_network_error_retried():
    calls = []

    def post(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise requests.ConnectionError("boom")
        return FakeResponse(200, _ok_body())

    client = OpenRouterClient("key", post=post, sleep=lambda s: None)
    assert client.chat({}).attempts == 2


def test_error_inside_200_body():
    body = {"error": {"code": 400, "message": "provider rejected"}}
    client = OpenRouterClient("key", post=lambda *a, **k: FakeResponse(200, body), sleep=lambda s: None)
    with pytest.raises(OpenRouterError):
        client.chat({})


def test_api_key_is_not_in_headers_repr():
    client = OpenRouterClient("secret-key", post=lambda *a, **k: FakeResponse(200, _ok_body()))
    assert "secret-key" not in repr(client)
