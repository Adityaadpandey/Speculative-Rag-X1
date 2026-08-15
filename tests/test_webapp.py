from fastapi.testclient import TestClient

import webapp.main as webapp_main
from webapp.main import app

client = TestClient(app)


def test_get_index_returns_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_get_collections_returns_parsed_list_and_key_flags(monkeypatch):
    monkeypatch.setattr(
        webapp_main,
        "_COLLECTIONS",
        [{"language": "hi", "strategy": "native"}, {"language": "en", "strategy": "sliding_window"}],
    )
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    response = client.get("/api/collections")
    assert response.status_code == 200
    data = response.json()
    assert data["collections"] == [
        {"language": "hi", "strategy": "native"},
        {"language": "en", "strategy": "sliding_window"},
    ]
    assert data["sarvam_key_configured"] is True
    assert data["anthropic_key_configured"] is False


def test_parse_collection_name_recognizes_known_language_strategy_pairs():
    assert webapp_main._parse_collection_name("hi_native") == {"language": "hi", "strategy": "native"}
    assert webapp_main._parse_collection_name("ta_sliding_window") == {
        "language": "ta",
        "strategy": "sliding_window",
    }
    assert webapp_main._parse_collection_name("unknown_collection") is None
