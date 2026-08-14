import os
from pathlib import Path

import httpx
import pytest

from speculative_rag.stt import SarvamAPIKeyMissingError, transcribe_audio

FIXTURE = Path(__file__).parent / "fixtures" / "sample_hindi.wav"


def _response(status_code: int, json_body: dict | None = None) -> httpx.Response:
    request = httpx.Request("POST", "https://api.sarvam.ai/speech-to-text")
    return httpx.Response(status_code=status_code, json=json_body, request=request)


def test_transcribe_audio_missing_api_key_raises(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    with pytest.raises(SarvamAPIKeyMissingError):
        transcribe_audio(FIXTURE, "hi-IN")


def test_transcribe_audio_success_first_attempt(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    calls = []

    def fake_post(url, *, headers, data, files, timeout):
        calls.append((url, headers, data))
        return _response(200, {"transcript": "नमस्ते"})

    monkeypatch.setattr(httpx, "post", fake_post)
    result = transcribe_audio(FIXTURE, "hi-IN")

    assert result == "नमस्ते"
    assert len(calls) == 1
    url, headers, data = calls[0]
    assert headers["api-subscription-key"] == "test-key"
    assert data["language_code"] == "hi-IN"


def test_transcribe_audio_retries_on_503_then_succeeds(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    monkeypatch.setattr("speculative_rag.stt.time.sleep", lambda _seconds: None)
    responses = [_response(503), _response(200, {"transcript": "ok"})]

    def fake_post(url, *, headers, data, files, timeout):
        return responses.pop(0)

    monkeypatch.setattr(httpx, "post", fake_post)
    result = transcribe_audio(FIXTURE, "hi-IN")

    assert result == "ok"
    assert responses == []


def test_transcribe_audio_raises_after_exhausting_retries(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    monkeypatch.setattr("speculative_rag.stt.time.sleep", lambda _seconds: None)
    call_count = {"n": 0}

    def fake_post(url, *, headers, data, files, timeout):
        call_count["n"] += 1
        return _response(503)

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(httpx.HTTPStatusError):
        transcribe_audio(FIXTURE, "hi-IN")
    assert call_count["n"] == 2


def test_transcribe_audio_raises_immediately_on_400_no_retry(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    call_count = {"n": 0}

    def fake_post(url, *, headers, data, files, timeout):
        call_count["n"] += 1
        return _response(400, {"error": "bad request"})

    monkeypatch.setattr(httpx, "post", fake_post)
    with pytest.raises(httpx.HTTPStatusError):
        transcribe_audio(FIXTURE, "hi-IN")
    assert call_count["n"] == 1


@pytest.mark.slow
def test_transcribe_audio_real_call():
    try:
        transcript = transcribe_audio(FIXTURE, "hi-IN")
    except SarvamAPIKeyMissingError:
        pytest.skip("SARVAM_API_KEY not set")
    assert isinstance(transcript, str)
    assert transcript.strip()
