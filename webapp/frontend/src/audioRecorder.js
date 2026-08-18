const MIME_CANDIDATES = ["audio/webm", "audio/ogg", "audio/mp4"];
const EXTENSION_BY_MIME_TYPE = { "audio/webm": "webm", "audio/ogg": "ogg", "audio/mp4": "m4a" };
const RECORD_SECONDS = 30;
const BAR_COUNT = 24;

export function createAudioRecorder({ waveformEl, countdownEl, onStop, onError }) {
  let mediaRecorder = null;
  let recordedChunks = [];
  let countdownTimer = null;
  let audioContext = null;
  let analyser = null;
  let rafId = null;
  let bars = [];

  function buildBars() {
    if (!waveformEl) return;
    waveformEl.innerHTML = "";
    bars = [];
    for (let i = 0; i < BAR_COUNT; i++) {
      const bar = document.createElement("div");
      bar.className = "bar";
      waveformEl.appendChild(bar);
      bars.push(bar);
    }
  }

  function animateWaveform() {
    if (!waveformEl) return;
    const data = new Uint8Array(analyser.frequencyBinCount);
    analyser.getByteFrequencyData(data);
    const step = Math.floor(data.length / BAR_COUNT) || 1;
    for (let i = 0; i < BAR_COUNT; i++) {
      const value = data[i * step] ?? 0;
      const pct = Math.max(8, Math.round((value / 255) * 100));
      bars[i].style.height = `${pct}%`;
    }
    rafId = requestAnimationFrame(animateWaveform);
  }

  function stopWaveform() {
    if (rafId != null) cancelAnimationFrame(rafId);
    rafId = null;
    if (audioContext) {
      audioContext.close();
      audioContext = null;
    }
    if (waveformEl) {
      waveformEl.hidden = true;
      waveformEl.innerHTML = "";
    }
  }

  async function start() {
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      onError(err);
      return;
    }

    recordedChunks = [];
    const supportedMimeType = MIME_CANDIDATES.find(
      (candidate) => typeof MediaRecorder.isTypeSupported === "function" && MediaRecorder.isTypeSupported(candidate)
    );
    mediaRecorder = supportedMimeType ? new MediaRecorder(stream, { mimeType: supportedMimeType }) : new MediaRecorder(stream);

    mediaRecorder.addEventListener("dataavailable", (e) => {
      if (e.data.size > 0) recordedChunks.push(e.data);
    });

    mediaRecorder.addEventListener("stop", () => {
      clearInterval(countdownTimer);
      countdownEl.textContent = "";
      stopWaveform();
      stream.getTracks().forEach((track) => track.stop());
      const actualMimeType = mediaRecorder.mimeType || "audio/webm";
      const baseMimeType = actualMimeType.split(";")[0].trim();
      const extension = EXTENSION_BY_MIME_TYPE[baseMimeType] ?? "webm";
      const blob = new Blob(recordedChunks, { type: actualMimeType });
      onStop(blob, `recording.${extension}`);
    });

    if (waveformEl) {
      audioContext = new (window.AudioContext || window.webkitAudioContext)();
      const source = audioContext.createMediaStreamSource(stream);
      analyser = audioContext.createAnalyser();
      analyser.fftSize = 128;
      source.connect(analyser);
      buildBars();
      waveformEl.hidden = false;
      animateWaveform();
    }

    mediaRecorder.start();

    let secondsLeft = RECORD_SECONDS;
    countdownEl.textContent = `${secondsLeft}s`;
    countdownTimer = setInterval(() => {
      secondsLeft -= 1;
      countdownEl.textContent = `${secondsLeft}s`;
      if (secondsLeft <= 0) {
        mediaRecorder.stop();
      }
    }, 1000);
  }

  function stop() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder.stop();
    }
  }

  function isRecording() {
    return mediaRecorder != null && mediaRecorder.state === "recording";
  }

  return { start, stop, isRecording };
}
