const state = {
  sarvamKeyConfigured: false,
  anthropicKeyConfigured: false,
};

let mediaRecorder = null;
let recordedChunks = [];
let recordTimer = null;

const languageSelect = document.getElementById("language-select");
const strategySelect = document.getElementById("strategy-select");
const queryTextInput = document.getElementById("query-text");
const askBtn = document.getElementById("ask-btn");
const audioFileInput = document.getElementById("audio-file");
const recordBtn = document.getElementById("record-btn");
const recordCountdown = document.getElementById("record-countdown");
const audioDisabledNote = document.getElementById("audio-disabled-note");
const resultPanel = document.getElementById("result-panel");
const outcomeBadge = document.getElementById("outcome-badge");
const resolvingNote = document.getElementById("resolving-note");
const resolvingStage = document.getElementById("resolving-stage");
const resolvingElapsed = document.getElementById("resolving-elapsed");
const answerBlock = document.getElementById("answer-block");
const answerText = document.getElementById("answer-text");
const answerSourceIds = document.getElementById("answer-source-ids");
const refusalText = document.getElementById("refusal-text");
const resultReason = document.getElementById("result-reason");
const latencyTableBody = document.querySelector("#latency-table tbody");
const totalE2eLatencyMs = document.getElementById("total-e2e-latency-ms");
const reviewPanel = document.getElementById("review-panel");
const reviewInitialValid = document.getElementById("review-initial-valid");
const reviewInitialAnswerable = document.getElementById("review-initial-answerable");
const rerankPanel = document.getElementById("rerank-panel");
const rerankLatencyMs = document.getElementById("rerank-latency-ms");
const rerankSelectedSources = document.getElementById("rerank-selected-sources");
const rerankAnswerGenerated = document.getElementById("rerank-answer-generated");
const rerankRegeneratedValid = document.getElementById("rerank-regenerated-valid");
const rerankRegeneratedAnswerable = document.getElementById("rerank-regenerated-answerable");
const synthesisPanel = document.getElementById("synthesis-panel");
const synthesisNote = document.getElementById("synthesis-note");
const errorBanner = document.getElementById("error-banner");

// Keyed by FinalAnswerStatus, the status of the FINAL validated answer --
// never the raw fast-path outcome, which can look "answered" from the
// extractive hot path alone while still being ungrounded or off-target.
const FINAL_STATUS_LABELS = {
  answered: "ANSWERED",
  not_answered: "NOT ANSWERED",
  ungrounded: "UNGROUNDED",
  insufficient_evidence: "INSUFFICIENT EVIDENCE",
  fallback: "FALLBACK — NO VALIDATED ANSWER",
};

// Keyed by the "stage" field answer_orchestrator.resolve_recovery reports
// via on_stage(), surfaced live while GET /api/answer/{request_id} is still
// "resolving" so the wait doesn't read as frozen.
const STAGE_LABELS = {
  starting: "validating and recovering in the background",
  validating: "validating the fast answer",
  reranking: "reranking passages and regenerating an answer",
  verifying: "verifying the regenerated answer is grounded",
  synthesizing: "synthesizing an answer across multiple passages",
};

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.hidden = false;
}

function clearError() {
  errorBanner.hidden = true;
  errorBanner.textContent = "";
}

async function loadCollections() {
  const response = await fetch("/api/collections");
  const data = await response.json();
  state.sarvamKeyConfigured = data.sarvam_key_configured;
  state.anthropicKeyConfigured = data.anthropic_key_configured;

  const languages = [...new Set(data.collections.map((c) => c.language))];
  const strategiesByLanguage = {};
  for (const c of data.collections) {
    (strategiesByLanguage[c.language] ??= []).push(c.strategy);
  }

  languageSelect.innerHTML = "";
  for (const language of languages) {
    const opt = document.createElement("option");
    opt.value = language;
    opt.textContent = language;
    languageSelect.appendChild(opt);
  }
  languageSelect.value = languages.includes("en") ? "en" : (languages[0] ?? "");

  function refreshStrategies() {
    const strategies = strategiesByLanguage[languageSelect.value] ?? [];
    strategySelect.innerHTML = "";
    for (const strategy of strategies) {
      const opt = document.createElement("option");
      opt.value = strategy;
      opt.textContent = strategy;
      strategySelect.appendChild(opt);
    }
    strategySelect.value = strategies.includes("native") ? "native" : (strategies[0] ?? "");
  }
  languageSelect.addEventListener("change", refreshStrategies);
  refreshStrategies();

  if (!state.sarvamKeyConfigured) {
    audioFileInput.disabled = true;
    recordBtn.disabled = true;
    audioDisabledNote.hidden = false;
  }
}

function resetPanels() {
  clearError();
  resultPanel.hidden = true;
  reviewPanel.hidden = true;
  rerankPanel.hidden = true;
  synthesisPanel.hidden = true;
  answerBlock.hidden = true;
  refusalText.hidden = true;
  latencyTableBody.innerHTML = "";
}

function renderLatencyTable(pipeline) {
  latencyTableBody.innerHTML = "";
  for (const timing of pipeline.timings) {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${timing.stage}</td><td>${timing.elapsed_ms.toFixed(2)} ms</td>`;
    latencyTableBody.appendChild(row);
  }
  const totalRow = document.createElement("tr");
  totalRow.innerHTML = `<td><strong>fast path TOTAL</strong></td><td><strong>${pipeline.total_ms.toFixed(2)} ms</strong></td>`;
  latencyTableBody.appendChild(totalRow);
}

// Shown the moment POST /api/answer returns (within the ~200ms latency
// budget) -- the raw extractive answer, not yet validated. Recovery
// (validation/rerank/regenerate/synthesis) is still running in the
// background at this point.
function renderFastPath(data) {
  resultPanel.hidden = false;
  resolvingNote.hidden = false;
  outcomeBadge.hidden = true;
  resultReason.textContent = "";
  refusalText.hidden = true;

  if (data.fast_answer) {
    answerBlock.hidden = false;
    answerText.textContent = data.fast_answer.text;
    answerSourceIds.textContent = data.fast_answer.source_passage_ids.join(", ") || "none";
  } else {
    answerBlock.hidden = true;
  }

  renderLatencyTable(data.pipeline);
  totalE2eLatencyMs.textContent = "…";
  resolvingStage.textContent = STAGE_LABELS.starting;
  resolvingElapsed.textContent = "";
}

// Shown once GET /api/answer/{request_id} reports status "done" -- the
// validated final answer. This is the only point at which final_status can
// legitimately be "answered".
function renderResult(data) {
  resultPanel.hidden = false;
  resolvingNote.hidden = true;
  outcomeBadge.hidden = false;
  outcomeBadge.textContent = FINAL_STATUS_LABELS[data.final_status] ?? data.final_status;
  outcomeBadge.className = `badge badge-${data.final_status}`;
  resultReason.textContent = data.reason ?? "";

  if (data.final_status === "answered" && data.answer) {
    answerBlock.hidden = false;
    refusalText.hidden = true;
    answerText.textContent = data.answer.text;
    answerSourceIds.textContent = data.answer.source_passage_ids.join(", ") || "none";
  } else {
    answerBlock.hidden = true;
    refusalText.hidden = false;
    refusalText.textContent = data.reason ?? "No validated answer could be produced.";
  }

  renderLatencyTable(data.pipeline);
  totalE2eLatencyMs.textContent = data.instrumentation.recovery_latency_ms.toFixed(2);

  renderInstrumentation(data.instrumentation);
}

function renderInstrumentation(instrumentation) {
  reviewPanel.hidden = false;
  reviewInitialValid.textContent = instrumentation.initial_answer_valid;
  reviewInitialAnswerable.textContent = instrumentation.initial_answerable;

  rerankPanel.hidden = !instrumentation.rerank_triggered;
  if (instrumentation.rerank_triggered) {
    rerankLatencyMs.textContent = (instrumentation.rerank_latency_ms ?? 0).toFixed(2);
    rerankSelectedSources.textContent = instrumentation.selected_source_ids.join(", ") || "none";
    rerankAnswerGenerated.textContent = instrumentation.reranked_answer_generated;
    rerankRegeneratedValid.textContent = instrumentation.regenerated_answer_valid;
    rerankRegeneratedAnswerable.textContent = instrumentation.regenerated_answerable;
  }

  synthesisPanel.hidden = !instrumentation.synthesis_triggered;
  if (instrumentation.synthesis_triggered) {
    synthesisNote.textContent =
      instrumentation.regenerated_answer_valid === false
        ? "Regeneration from the reranked evidence failed validation, so multi-passage synthesis ran as a deeper fallback."
        : "Multi-passage synthesis ran as part of answer recovery.";
  }
}

const POLL_INTERVAL_MS = 700;

async function pollForFinalAnswer(requestId) {
  const pollStart = performance.now();
  for (;;) {
    let response;
    let data;
    try {
      response = await fetch(`/api/answer/${requestId}`);
      data = await response.json();
    } catch (err) {
      showError(`polling for result failed: ${err.message}`);
      return;
    }
    if (!response.ok) {
      showError(data.error ?? "polling for result failed");
      return;
    }
    if (data.status === "done") {
      renderResult(data);
      return;
    }
    resolvingStage.textContent = STAGE_LABELS[data.stage] ?? STAGE_LABELS.starting;
    resolvingElapsed.textContent = `${((performance.now() - pollStart) / 1000).toFixed(1)}s`;
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }
}

async function runQuery(queryText) {
  resetPanels();
  if (!queryText.trim()) {
    return;
  }

  resultPanel.hidden = false;
  resolvingNote.hidden = false;
  outcomeBadge.hidden = true;

  let response;
  let data;
  try {
    response = await fetch("/api/answer", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query_text: queryText,
        language: languageSelect.value,
        strategy: strategySelect.value,
      }),
    });
    data = await response.json();
  } catch (err) {
    showError(`query failed: ${err.message}`);
    resultPanel.hidden = true;
    return;
  }
  if (!response.ok) {
    showError(data.error ?? "query failed");
    resultPanel.hidden = true;
    return;
  }

  renderFastPath(data);
  await pollForFinalAnswer(data.request_id);
}

askBtn.addEventListener("click", () => {
  runQuery(queryTextInput.value);
});
queryTextInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    runQuery(queryTextInput.value);
  }
});

async function transcribeAndQuery(blob, filename) {
  resetPanels();
  const formData = new FormData();
  formData.append("file", blob, filename);
  formData.append("language", languageSelect.value);

  let response;
  let data;
  try {
    response = await fetch("/api/transcribe", { method: "POST", body: formData });
    data = await response.json();
  } catch (err) {
    showError(`transcription failed: ${err.message}`);
    return;
  }
  if (!response.ok) {
    showError(data.error ?? "transcription failed");
    return;
  }
  queryTextInput.value = data.transcript;
  runQuery(data.transcript);
}

audioFileInput.addEventListener("change", () => {
  const file = audioFileInput.files[0];
  if (file) {
    transcribeAndQuery(file, file.name);
  }
});

recordBtn.addEventListener("click", async () => {
  if (mediaRecorder && mediaRecorder.state === "recording") {
    mediaRecorder.stop();
    return;
  }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    recordedChunks = [];
    const mimeCandidates = ["audio/webm", "audio/ogg", "audio/mp4"];
    const supportedMimeType = mimeCandidates.find(
      (candidate) => typeof MediaRecorder.isTypeSupported === "function" && MediaRecorder.isTypeSupported(candidate)
    );
    mediaRecorder = supportedMimeType
      ? new MediaRecorder(stream, { mimeType: supportedMimeType })
      : new MediaRecorder(stream);
    mediaRecorder.addEventListener("dataavailable", (e) => {
      if (e.data.size > 0) {
        recordedChunks.push(e.data);
      }
    });
    mediaRecorder.addEventListener("stop", () => {
      clearInterval(recordTimer);
      recordCountdown.textContent = "";
      recordBtn.textContent = "Record";
      stream.getTracks().forEach((track) => track.stop());
      const actualMimeType = mediaRecorder.mimeType || "audio/webm";
      const extensionByMimeType = {
        "audio/webm": "webm",
        "audio/ogg": "ogg",
        "audio/mp4": "m4a",
      };
      const baseMimeType = actualMimeType.split(";")[0].trim();
      const extension = extensionByMimeType[baseMimeType] ?? "webm";
      const blob = new Blob(recordedChunks, { type: actualMimeType });
      transcribeAndQuery(blob, `recording.${extension}`);
    });
    mediaRecorder.start();
    recordBtn.textContent = "Stop";

    let secondsLeft = 30;
    recordCountdown.textContent = `${secondsLeft}s`;
    recordTimer = setInterval(() => {
      secondsLeft -= 1;
      recordCountdown.textContent = `${secondsLeft}s`;
      if (secondsLeft <= 0) {
        mediaRecorder.stop();
      }
    }, 1000);
  } catch (err) {
    showError(`microphone access failed: ${err.message}`);
  }
});

loadCollections().catch((err) => showError(`failed to load collections: ${err.message}`));
