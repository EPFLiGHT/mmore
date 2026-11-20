import json
from io import BytesIO
from unittest.mock import patch

from mmore.websearchRAG.websearch import WebsearchOnly


class _FakeResponse:
    def __init__(self, payload, status=200):
        self.status = status
        self._payload = json.dumps(payload).encode("utf-8")

    def read(self):
        return self._payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_keenable_keyless_maps_results_and_sends_title_header():
    searcher = WebsearchOnly(provider="keenable", max_results=3)
    payload = {
        "results": [
            {
                "title": "A",
                "url": "https://a.example/page",
                "description": "short",
                "snippet": "long",
            },
            {"title": "No URL", "description": "drop"},
            {"title": "B", "url": "https://b.example/page", "snippet": "fallback"},
        ]
    }
    with patch("mmore.websearchRAG.websearch.urlopen", return_value=_FakeResponse(payload)) as urlopen:
        results = searcher.websearch_pipeline("q")

    request = urlopen.call_args[0][0]
    assert request.full_url == "https://api.keenable.ai/v1/search/public"
    assert request.get_method() == "POST"
    header_map = {k.lower(): v for k, v in request.header_items()}
    assert header_map.get("x-keenable-title") == "mmore"
    assert "x-api-key" not in header_map
    body = json.loads(request.data.decode("utf-8"))
    assert body == {"query": "q", "max_results": 3}
    assert results == [
        {"body": "short", "href": "https://a.example/page", "title": "A"},
        {"body": "fallback", "href": "https://b.example/page", "title": "B"},
    ]


def test_keenable_uses_keyed_endpoint_when_api_key_is_set(monkeypatch):
    monkeypatch.setenv("KEENABLE_API_KEY", "secret")
    searcher = WebsearchOnly(provider="keenable", max_results=2)
    with patch(
        "mmore.websearchRAG.websearch.urlopen",
        return_value=_FakeResponse({"results": []}),
    ) as urlopen:
        searcher.websearch_pipeline("q")
    request = urlopen.call_args[0][0]
    assert request.full_url == "https://api.keenable.ai/v1/search"
    header_map = {k.lower(): v for k, v in request.header_items()}
    assert header_map.get("x-api-key") == "secret"
