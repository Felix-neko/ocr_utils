"""Клиент OpenRouter поверх SDK: адаптер payload, ретраи, разбор usage и ошибок — без сети (httpx.MockTransport)."""

import json

import httpx
import pytest

from ocr_utils.external_ocr_services.client import OpenRouterClient, OpenRouterError, adapt_reasoning


def _ok_body(text="{}", cost=0.0021):
    """Ответ в форме, которую разбирает модель SDK: choices, usage, metadata с выбранным endpoint."""
    return {
        "id": "gen-1",
        "object": "chat.completion",
        "created": 1,
        "model": "deepseek/deepseek-v4.1-flash",
        "system_fingerprint": None,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop", "logprobs": None}
        ],
        "usage": {
            "prompt_tokens": 1500,
            "completion_tokens": 700,
            "total_tokens": 2200,
            "cost": cost,
            "completion_tokens_details": {"reasoning_tokens": 12},
            "prompt_tokens_details": {"cached_tokens": 1200},
        },
        "openrouter_metadata": {
            "attempt": 1,
            "is_byok": False,
            "region": None,
            "requested": "deepseek/deepseek-v4.1-flash",
            "strategy": "default",
            "summary": "ok",
            "endpoints": {
                "total": 2,
                "available": [
                    {"model": "deepseek/deepseek-v4.1-flash", "provider": "DeepSeek", "selected": True},
                    {"model": "deepseek/deepseek-v4.1-flash", "provider": "Relace", "selected": False},
                ],
            },
        },
    }


class Server:
    """Подменный сервер: очередь ответов (или исключений) по порядку вызовов, запись запросов."""

    def __init__(self, *replies):
        self.replies = list(replies)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        reply = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        if isinstance(reply, Exception):
            raise reply
        status, body = reply
        if isinstance(body, dict):
            return httpx.Response(status, json=body)
        return httpx.Response(status, text=body, headers={"content-type": "text/plain"})

    def client(self, **kwargs) -> OpenRouterClient:
        http = httpx.Client(transport=httpx.MockTransport(self))
        kwargs.setdefault("sleep", lambda s: None)
        return OpenRouterClient("key", http_client=http, **kwargs)

    def sent(self, index=-1) -> dict:
        return json.loads(self.requests[index].content)


PAYLOAD = {
    "model": "deepseek/deepseek-v4.1-flash",
    "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    "temperature": 0,
    "max_tokens": 10,
    "provider": {"order": ["deepseek"], "allow_fallbacks": True, "ignore": ["relace"]},
    "response_format": {"type": "json_object"},
    "reasoning": {"enabled": False},
}


def test_parses_usage_cost_and_provider_from_metadata():
    server = Server((200, _ok_body()))
    response = server.client().chat(PAYLOAD)
    assert response.text == "{}"
    assert response.cost_usd == pytest.approx(0.0021)
    assert (response.prompt_tokens, response.completion_tokens, response.reasoning_tokens) == (1500, 700, 12)
    assert response.cached_tokens == 1200
    assert response.provider == "DeepSeek" and response.finish_reason == "stop" and response.attempts == 1
    assert response.request_id == "gen-1" and response.raw_usage["cost"] == pytest.approx(0.0021)


def test_request_carries_payload_metadata_header_and_effort_none():
    server = Server((200, _ok_body()))
    server.client().chat(PAYLOAD)
    request = server.requests[0]
    assert request.url.path.endswith("/chat/completions")
    assert request.headers["authorization"] == "Bearer key"
    assert request.headers["x-openrouter-metadata"] == "enabled"
    sent = server.sent()
    assert sent["reasoning"] == {"effort": "none"}  # enabled:false SDK не знает — переведено
    assert sent["provider"] == {"order": ["deepseek"], "allow_fallbacks": True, "ignore": ["relace"]}
    assert sent["response_format"] == {"type": "json_object"} and sent["temperature"] == 0 and sent["max_tokens"] == 10
    assert sent["messages"][0]["content"][0]["text"] == "hi"


def test_adapt_reasoning():
    assert adapt_reasoning(None) is None
    assert adapt_reasoning({"enabled": False}) == {"effort": "none"}
    assert adapt_reasoning({"effort": "low"}) == {"effort": "low"}
    assert adapt_reasoning({"max_tokens": 1024}) is None  # потолок через SDK не передать
    assert adapt_reasoning({"enabled": True}) is None


def test_unknown_payload_field_is_refused():
    with pytest.raises(OpenRouterError):
        Server((200, _ok_body())).client().chat({**PAYLOAD, "tools": []})


def test_retries_on_429_then_succeeds():
    server = Server((429, {"error": {"message": "slow down"}}), (429, "slow down"), (200, _ok_body()))
    slept = []
    response = server.client(attempts=5, sleep=slept.append).chat(PAYLOAD)
    assert response.attempts == 3 and len(slept) == 2 and len(server.requests) == 3


def test_gives_up_after_attempts():
    server = Server((503, "down"))
    with pytest.raises(OpenRouterError) as error:
        server.client(attempts=2).chat(PAYLOAD)
    assert error.value.status == 503 and len(server.requests) == 2


def test_400_is_not_retried():
    server = Server((400, {"error": {"message": "bad response_format", "code": 400}}))
    with pytest.raises(OpenRouterError) as error:
        server.client(attempts=5).chat(PAYLOAD)
    assert error.value.status == 400 and "response_format" in error.value.body and len(server.requests) == 1


def test_network_error_retried():
    server = Server(httpx.ConnectError("boom"), (200, _ok_body()))
    assert server.client().chat(PAYLOAD).attempts == 2


def test_error_inside_200_body():
    server = Server((200, {"error": {"code": 400, "message": "provider rejected"}}))
    with pytest.raises(OpenRouterError) as error:
        server.client(attempts=3).chat(PAYLOAD)
    assert error.value.status == 400 and "provider rejected" in str(error.value) and len(server.requests) == 1


def test_error_inside_200_body_retried_on_429():
    server = Server((200, {"error": {"code": 429, "message": "rate"}}), (200, _ok_body()))
    assert server.client().chat(PAYLOAD).attempts == 2


def test_content_as_parts_and_missing_usage():
    body = _ok_body()
    body["choices"][0]["message"]["content"] = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]
    del body["usage"]
    response = Server((200, body)).client().chat(PAYLOAD)
    assert response.text == "ab" and response.cost_usd is None and response.prompt_tokens == 0


def test_api_key_is_not_in_repr():
    client = Server((200, _ok_body())).client()
    assert "key" not in repr(client).replace("OpenRouterClient", "")
