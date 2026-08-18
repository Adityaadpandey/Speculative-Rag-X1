import speculative_rag.config as config_module


class _RecordingQdrantClient:
    last_kwargs: dict | None = None

    def __init__(self, **kwargs):
        _RecordingQdrantClient.last_kwargs = kwargs


def test_urdu_is_configured_for_ingestion_and_stt():
    urdu = next(source for source in config_module.LANGUAGE_SOURCES if source.language == "ur")

    assert urdu.parquet_path.name == "urdval.parquet"
    assert urdu.text_field == "Translated_passages"
    assert config_module.BCP47_LANGUAGE_CODES["ur"] == "ur-IN"


def test_get_qdrant_client_remote_uses_rest_with_generous_timeout(monkeypatch):
    monkeypatch.setenv("QDRANT_URL", "https://example.cloud.qdrant.io")
    monkeypatch.setenv("QDRANT_API_KEY", "test-key")
    monkeypatch.setattr(config_module, "QdrantClient", _RecordingQdrantClient)

    config_module.get_qdrant_client()

    assert _RecordingQdrantClient.last_kwargs == {
        "url": "https://example.cloud.qdrant.io",
        "api_key": "test-key",
        "timeout": 60,
    }


def test_get_qdrant_client_local_when_url_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    local_path = tmp_path / "qdrant_local"
    monkeypatch.setattr(config_module.SETTINGS, "qdrant_path", local_path)
    monkeypatch.setattr(config_module, "QdrantClient", _RecordingQdrantClient)

    config_module.get_qdrant_client()

    assert _RecordingQdrantClient.last_kwargs == {"path": str(local_path)}
