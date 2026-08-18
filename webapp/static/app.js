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
const answerText = document.getElementById("answer-text");
const answerSourceIds = document.getElementById("answer-source-ids");
const answerSourcesRow = document.getElementById("answer-sources-row");
const refusalText = document.getElementById("refusal-text");
const resultReason = document.getElementById("result-reason");
const latencyTableBody = document.querySelector("#latency-table tbody");
const totalE2eLatencyMs = document.getElementById("total-e2e-latency-ms");
const errorBanner = document.getElementById("error-banner");

// Every LLM call in the recovery chain (validate / rerank+regenerate /
// synthesize) runs on Claude Haiku (see src/speculative_rag/reflect.py
// _MODEL) -- the fast path above it is pure BM25/embedding retrieval plus
// extractive span selection and never touches an LLM. The "no LLM" / "Claude
// Haiku" tags on each timeline step are static since which stage uses an
// LLM is fixed by the pipeline shape, not something that changes per request.
const steps = {
  fast: {
    el: document.querySelector('[data-step="fast"]'),
    detail: document.getElementById("step-fast-detail"),
    time: document.getElementById("step-fast-time"),
  },
  verify: {
    el: document.querySelector('[data-step="verify"]'),
    detail: document.getElementById("step-verify-detail"),
    time: document.getElementById("step-verify-time"),
  },
  rerank: {
    el: document.querySelector('[data-step="rerank"]'),
    detail: document.getElementById("step-rerank-detail"),
    time: document.getElementById("step-rerank-time"),
  },
  synthesis: {
    el: document.querySelector('[data-step="synthesis"]'),
    detail: document.getElementById("step-synthesis-detail"),
    time: document.getElementById("step-synthesis-time"),
  },
};

function setStep(name, { state: stepState, detail, time, hidden } = {}) {
  const step = steps[name];
  if (hidden !== undefined) step.el.hidden = hidden;
  if (stepState) {
    step.el.classList.remove("active", "done", "corrected");
    step.el.classList.add(stepState);
  }
  if (detail !== undefined) step.detail.textContent = detail;
  if (time !== undefined) step.time.textContent = time;
}

// Final-status framing: an ungrounded/insufficient-evidence/fallback result
// is the safety net working as designed (the pipeline declining to guess),
// not a system failure -- so only a genuinely verified answer gets the
// "good" (green) treatment. Everything else reads as an amber, informative
// outcome instead of a red "false".
const STATUS_META = {
  answered: { label: "Verified Answer", cls: "good" },
  not_answered: { label: "No Direct Answer Found", cls: "warn" },
  insufficient_evidence: { label: "Insufficient Evidence", cls: "warn" },
  ungrounded: { label: "Answer Withheld — Not Grounded", cls: "warn" },
  fallback: { label: "No Verified Answer After Full Recovery", cls: "warn" },
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
  outcomeBadge.hidden = true;
  answerText.hidden = true;
  refusalText.hidden = true;
  resultReason.textContent = "";
  latencyTableBody.innerHTML = "";
  for (const name of Object.keys(steps)) {
    setStep(name, { state: "pending", detail: "", time: "" });
  }
  steps.rerank.el.hidden = true;
  steps.synthesis.el.hidden = true;
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

function renderAnswerBlock(text, sourceIds) {
  answerText.hidden = false;
  refusalText.hidden = true;
  answerText.textContent = text;
  answerSourcesRow.hidden = false;
  answerSourceIds.textContent = sourceIds.join(", ") || "none";
}

function renderRefusalBlock(reason) {
  answerText.hidden = true;
  refusalText.hidden = false;
  answerSourcesRow.hidden = true;
  refusalText.textContent = reason || "No validated answer could be produced.";
}

// Whether check_groundedness_llm will run on the fast-path answer at all --
// it only gets invoked when the fast path itself produced
// RagOutcome.ANSWERED (see answer_orchestrator.resolve_recovery); for any
// other fast-path outcome, recovery skips straight to rerank (or ends
// immediately), so the "verify" timeline step doesn't apply and needs to
// stay hidden instead of implying a check that never happened.
function willRunVerify(pipelineOutcome) {
  return pipelineOutcome === "answered";
}

// Shown the moment POST /api/answer returns (within the ~200ms latency
// budget) -- the raw extractive answer, not yet validated. Recovery
// (validation/rerank/regenerate/synthesis) is still running in the
// background at this point.
function renderFastPath(data) {
  resultPanel.hidden = false;
  outcomeBadge.hidden = true;

  const totalMs = data.pipeline.total_ms.toFixed(2);
  if (data.fast_answer) {
    renderAnswerBlock(data.fast_answer.text, data.fast_answer.source_passage_ids);
    setStep("fast", {
      state: "done",
      detail: `extractive answer found in ${totalMs}ms — verifying now`,
      time: `${totalMs} ms`,
    });
  } else {
    answerText.hidden = true;
    answerSourcesRow.hidden = true;
    refusalText.hidden = false;
    refusalText.textContent = "Checking the retrieved passages for an answer…";
    setStep("fast", {
      state: "done",
      detail: `no confident extractive span in ${totalMs}ms — checking retrieved passages more deeply`,
      time: `${totalMs} ms`,
    });
  }

  renderLatencyTable(data.pipeline);
  totalE2eLatencyMs.textContent = "…";

  if (willRunVerify(data.pipeline.outcome)) {
    setStep("verify", {
      state: "active",
      hidden: false,
      detail: "Claude Haiku is checking the answer is grounded, complete, and on-target…",
      time: "",
    });
  } else {
    setStep("verify", { hidden: true });
    setStep("rerank", {
      state: "active",
      hidden: false,
      detail: "Claude Haiku is searching every retrieved passage for the evidence…",
      time: "",
    });
  }
}

// Live progress while GET /api/answer/{request_id} is still "resolving" --
// stage is one of starting/validating/reranking/verifying/synthesizing, per
// answer_orchestrator.resolve_recovery's on_stage() calls. "verifying" fires
// twice (after rerank+regenerate, and after synthesis) so it's routed to
// whichever of those two steps most recently became active.
let currentRecoveryPhase = null;

function renderStageProgress(stage) {
  if (stage === "reranking") {
    currentRecoveryPhase = "rerank";
    setStep("rerank", {
      state: "active",
      hidden: false,
      detail: "Claude Haiku is re-ranking retrieved passages and regenerating an answer…",
    });
    if (!steps.verify.el.hidden) {
      setStep("verify", { state: "corrected", detail: "flagged for a closer look — escalating to re-ranking" });
    }
  } else if (stage === "synthesizing") {
    currentRecoveryPhase = "synthesis";
    setStep("synthesis", {
      state: "active",
      hidden: false,
      detail: "Claude Haiku is synthesizing an answer across multiple passages…",
    });
    setStep("rerank", { state: "corrected", detail: "regenerated answer needed more support — escalating to synthesis" });
  } else if (stage === "verifying" && currentRecoveryPhase) {
    setStep(currentRecoveryPhase, { detail: "double-checking the new answer is grounded before returning it…" });
  }
}

// Shown once GET /api/answer/{request_id} reports status "done" -- the
// validated final answer. This is the only point at which final_status can
// legitimately be "answered".
function renderResult(data) {
  resultPanel.hidden = false;
  outcomeBadge.hidden = false;
  const meta = STATUS_META[data.final_status] ?? { label: data.final_status, cls: "warn" };
  outcomeBadge.textContent = meta.label;
  outcomeBadge.className = `badge badge-${meta.cls}`;
  resultReason.textContent = data.reason ?? "";

  if (data.final_status === "answered" && data.answer) {
    renderAnswerBlock(data.answer.text, data.answer.source_passage_ids);
  } else {
    renderRefusalBlock(data.reason);
  }

  renderLatencyTable(data.pipeline);
  totalE2eLatencyMs.textContent = data.instrumentation.recovery_latency_ms.toFixed(2);

  renderInstrumentation(data.instrumentation, data.final_status);
}

function renderInstrumentation(instrumentation, finalStatus) {
  if (!steps.verify.el.hidden) {
    if (instrumentation.initial_answer_valid) {
      setStep("verify", {
        state: "done",
        detail: "passed verification on the first pass — grounded, complete, and on-target.",
      });
    } else {
      setStep("verify", {
        state: "corrected",
        detail: "the fast answer needed a closer look, so the pipeline automatically escalated to re-ranking.",
      });
    }
  }

  if (instrumentation.rerank_triggered) {
    const rerankTime = instrumentation.rerank_latency_ms != null ? `${instrumentation.rerank_latency_ms.toFixed(0)} ms` : "";
    const sourceCount = instrumentation.selected_source_ids.length;
    if (!instrumentation.reranked_answer_generated || sourceCount === 0) {
      setStep("rerank", {
        state: "corrected",
        hidden: false,
        detail: "none of the retrieved passages contained the answer — declined rather than guess.",
        time: rerankTime,
      });
    } else if (instrumentation.regenerated_answer_valid) {
      setStep("rerank", {
        state: "done",
        hidden: false,
        detail: `selected ${sourceCount} source${sourceCount === 1 ? "" : "s"} from the full candidate pool and regenerated a verified answer.`,
        time: rerankTime,
      });
    } else {
      setStep("rerank", {
        state: "corrected",
        hidden: false,
        detail: `selected ${sourceCount} source${sourceCount === 1 ? "" : "s"}, but the regenerated answer needed further verification.`,
        time: rerankTime,
      });
    }
  }

  if (instrumentation.synthesis_triggered) {
    if (finalStatus === "answered") {
      setStep("synthesis", {
        state: "done",
        hidden: false,
        detail: "combined evidence across multiple passages to produce a verified answer.",
      });
    } else {
      setStep("synthesis", {
        state: "corrected",
        hidden: false,
        detail: "no combination of retrieved passages fully supported an answer — declined rather than guess.",
      });
    }
  }
}

const POLL_INTERVAL_MS = 700;

async function pollForFinalAnswer(requestId) {
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
    renderStageProgress(data.stage);
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }
}

async function runQuery(queryText) {
  resetPanels();
  currentRecoveryPhase = null;
  if (!queryText.trim()) {
    return;
  }

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
