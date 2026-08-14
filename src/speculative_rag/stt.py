"""Sarvam AI speech-to-text integration. Wraps the Sarvam speech-to-text
REST endpoint as a bounded-retry multipart POST. Takes an audio file path
as input only -- no live mic handling here, that's out of scope."""

import os
import time
from pathlib import Path

import httpx

from speculative_rag.config import SETTINGS

_RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}
_MAX_ATTEMPTS = 2
_BACKOFF_SECONDS = 1.0


class SarvamAPIKeyMissingError(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "SARVAM_API_KEY is not set in the environment. "
            "Set it before calling transcribe_audio()."
        )


def transcribe_audio(file_path: Path, language_code: str) -> str:
    api_key = os.environ.get("SARVAM_API_KEY")
    if not api_key:
        raise SarvamAPIKeyMissingError()

    headers = {"api-subscription-key": api_key}
    data = {"language_code": language_code}

    last_error: httpx.HTTPStatusError | None = None
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        with file_path.open("rb") as audio_file:
            files = {"file": (file_path.name, audio_file, "audio/wav")}
            response = httpx.post(
                SETTINGS.sarvam_stt_url,
                headers=headers,
                data=data,
                files=files,
                timeout=30.0,
            )

        if response.status_code not in _RETRYABLE_STATUS_CODES:
            response.raise_for_status()
            return response.json()["transcript"]

        last_error = httpx.HTTPStatusError(
            f"Sarvam STT returned {response.status_code} after {attempt} attempt(s)",
            request=response.request,
            response=response,
        )
        if attempt < _MAX_ATTEMPTS:
            time.sleep(_BACKOFF_SECONDS)

    assert last_error is not None
    raise last_error
