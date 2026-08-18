import "./styles/tokens.css";
import "./styles/main.css";
import { fetchCollections, postAnswer, getAnswer, transcribeAudio, ApiError } from "./api.js";
import { initToastContainer, showToast, clearToasts } from "./toast.js";
import { initGradientMesh } from "./gradientMesh.js";
import { initTimeline, setStepState, STEP_NAMES } from "./timeline.js";
import { renderAnswerDiff } from "./answerDiff.js";
import { createAudioRecorder } from "./audioRecorder.js";
import { collapseHero } from "./heroChoreography.js";

// Every LLM call in the recovery chain (validate / rerank+regenerate / synthesize) runs on
// Claude Haiku -- the fast path never touches an LLM. Unchanged from the pre-redesign app.js.
const STATUS_META = {
  answered: { label: "Verified Answer", cls: "good" },
  not_answered: { label: "No Direct Answer Found", cls: "warn" },
  insufficient_evidence: { label: "Insufficient Evidence", cls: "warn" },
  ungrounded: { label: "Answer Withheld — Not Grounded", cls: "warn" },
  fallback: { label: "No Verified Answer After Full Recovery", cls: "warn" },
};

const state = {
  sarvamKeyConfigured: false,
  anthropicKeyConfigured: false,
  collapsed: false,
  activeControls: "hero",
  currentRecoveryPhase: null,
  lastAnswerText: null,
};

const heroEl = document.getElementById("hero");
const compactBarEl = document.getElementById("compact-bar");
const resultViewEl = document.getElementById("result-view");
const gradientMeshEl = document.getElementById("gradient-mesh");
const toastContainerEl = document.getElementById("toast-container");
const timelineRootEl = document.querySelector('[data-role="timeline"]');

const answerCardEl = document.querySelector('[data-role="answer-card"]');
const outcomeBadgeEl = document.querySelector('[data-role="outcome-badge"]');
const answerSkeletonEl = document.querySelector('[data-role="answer-skeleton"]');
const answerTextEl = document.querySelector('[data-role="answer-text"]');
const refusalTextEl = document.querySelector('[data-role="refusal-text"]');
const answerSourcesRowEl = document.querySelector('[data-role="answer-sources-row"]');
const answerSourceIdsEl = document.querySelector('[data-role="answer-source-ids"]');
const resultReasonEl = document.querySelector('[data-role="result-reason"]');
const latencyTableBodyEl = document.querySelector('[data-role="latency-table-body"]');
const totalE2eLatencyMsEl = document.querySelector('[data-role="total-e2e-latency-ms"]');

const audioFileInputEl = heroEl.querySelector('[data-role="audio-file"]');
const recordBtnEl = heroEl.querySelector('[data-role="record-btn"]');
const waveformEl = heroEl.querySelector('[data-role="waveform"]');
const recordCountdownEl = heroEl.querySelector('[data-role="record-countdown"]');
const audioDisabledNoteEl = heroEl.querySelector('[data-role="audio-disabled-note"]');

initToastContainer(toastContainerEl);
initGradientMesh(gradientMeshEl);
const timelineHandle = initTimeline(timelineRootEl);

function buildControlSet(containerEl) {
  const set = {
    languageSelect: containerEl.querySelector('[data-role="language-select"]'),
    strategySelect: containerEl.querySelector('[data-role="strategy-select"]'),
    queryInput: containerEl.querySelector('[data-role="query-input"]'),
    askBtn: containerEl.querySelector('[data-role="ask-btn"]'),
  };
  set.askBtnLabel = set.askBtn.querySelector(".ask-btn-label");
  set.askBtnSpinner = set.askBtn.querySelector(".ask-btn-spinner");
  return set;
}

const heroControls = buildControlSet(heroEl);
const compactControls = buildControlSet(compactBarEl);
const controlSets = [heroControls, compactControls];

function activeControlSet() {
  return state.activeControls === "hero" ? heroControls : compactControls;
}

let strategiesByLanguage = {};

function populateSelect(selectEl, values, preferred) {
  selectEl.innerHTML = "";
  for (const value of values) {
    const opt = document.createElement("option");
    opt.value = value;
    opt.textContent = value;
    selectEl.appendChild(opt);
  }
  selectEl.value = values.includes(preferred) ? preferred : (values[0] ?? "");
}

function refreshStrategiesFor(controls) {
  const strategies = strategiesByLanguage[controls.languageSelect.value] ?? [];
  populateSelect(controls.strategySelect, strategies, "native");
}

async function loadCollections() {
  const data = await fetchCollections();
  state.sarvamKeyConfigured = data.sarvam_key_configured;
  state.anthropicKeyConfigured = data.anthropic_key_configured;

  const languages = [...new Set(data.collections.map((c) => c.language))];
  strategiesByLanguage = {};
  for (const c of data.collections) {
    (strategiesByLanguage[c.language] ??= []).push(c.strategy);
  }

  for (const controls of controlSets) {
    populateSelect(controls.languageSelect, languages, "en");
    refreshStrategiesFor(controls);
    controls.languageSelect.addEventListener("change", () => refreshStrategiesFor(controls));
  }

  if (!state.sarvamKeyConfigured) {
    audioFileInputEl.disabled = true;
    recordBtnEl.disabled = true;
    audioDisabledNoteEl.hidden = false;
  }
}

function resetResultView() {
  clearToasts();
  outcomeBadgeEl.hidden = true;
  answerSkeletonEl.hidden = false;
  answerTextEl.hidden = true;
  refusalTextEl.hidden = true;
  answerSourcesRowEl.hidden = true;
  resultReasonEl.textContent = "";
  latencyTableBodyEl.innerHTML = "";
  totalE2eLatencyMsEl.textContent = "";
  state.lastAnswerText = null;
  state.currentRecoveryPhase = null;
  for (const name of STEP_NAMES) {
    setStepState(timelineHandle, name, { state: "pending", detail: "", time: "" });
  }
  setStepState(timelineHandle, "rerank", { hidden: true });
  setStepState(timelineHandle, "synthesis", { hidden: true });
  setStepState(timelineHandle, "verify", { hidden: false });
}

function renderLatencyTable(pipeline) {
  latencyTableBodyEl.innerHTML = "";
  for (const timing of pipeline.timings) {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${timing.stage}</td><td>${timing.elapsed_ms.toFixed(2)} ms</td>`;
    latencyTableBodyEl.appendChild(row);
  }
  const totalRow = document.createElement("tr");
  totalRow.innerHTML = `<td><strong>fast path TOTAL</strong></td><td><strong>${pipeline.total_ms.toFixed(2)} ms</strong></td>`;
  latencyTableBodyEl.appendChild(totalRow);
}

function showAnswer(text, sourceIds) {
  answerSkeletonEl.hidden = true;
  refusalTextEl.hidden = true;
  answerTextEl.hidden = false;
  renderAnswerDiff(answerTextEl, state.lastAnswerText, text);
  state.lastAnswerText = text;
  answerSourcesRowEl.hidden = false;
  answerSourceIdsEl.textContent = sourceIds.join(", ") || "none";
}

function showRefusal(reason) {
  answerSkeletonEl.hidden = true;
  answerTextEl.hidden = true;
  answerSourcesRowEl.hidden = true;
  refusalTextEl.hidden = false;
  refusalTextEl.textContent = reason || "No validated answer could be produced.";
  state.lastAnswerText = null;
}

function willRunVerify(pipelineOutcome) {
  return pipelineOutcome === "answered";
}

function renderFastPath(data) {
  resultViewEl.hidden = false;
  const totalMs = data.pipeline.total_ms.toFixed(2);

  if (data.fast_answer) {
    showAnswer(data.fast_answer.text, data.fast_answer.source_passage_ids);
    setStepState(timelineHandle, "fast", {
      state: "done",
      detail: `extractive answer found in ${totalMs}ms — verifying now`,
      time: `${totalMs} ms`,
    });
  } else {
    showRefusal("Checking the retrieved passages for an answer…");
    setStepState(timelineHandle, "fast", {
      state: "done",
      detail: `no confident extractive span in ${totalMs}ms — checking retrieved passages more deeply`,
      time: `${totalMs} ms`,
    });
  }

  renderLatencyTable(data.pipeline);
  totalE2eLatencyMsEl.textContent = "…";

  if (willRunVerify(data.pipeline.outcome)) {
    setStepState(timelineHandle, "verify", {
      state: "active",
      hidden: false,
      detail: "Claude Haiku is checking the answer is grounded, complete, and on-target…",
    });
  } else {
    setStepState(timelineHandle, "verify", { hidden: true });
    setStepState(timelineHandle, "rerank", {
      state: "active",
      hidden: false,
      detail: "Claude Haiku is searching every retrieved passage for the evidence…",
    });
  }
}

function renderStageProgress(stage) {
  if (stage === "reranking") {
    state.currentRecoveryPhase = "rerank";
    setStepState(timelineHandle, "rerank", {
      state: "active",
      hidden: false,
      detail: "Claude Haiku is re-ranking retrieved passages and regenerating an answer…",
    });
    if (!timelineHandle.steps.verify.el.hidden) {
      setStepState(timelineHandle, "verify", { state: "corrected", detail: "flagged for a closer look — escalating to re-ranking" });
    }
  } else if (stage === "synthesizing") {
    state.currentRecoveryPhase = "synthesis";
    setStepState(timelineHandle, "synthesis", {
      state: "active",
      hidden: false,
      detail: "Claude Haiku is synthesizing an answer across multiple passages…",
    });
    setStepState(timelineHandle, "rerank", { state: "corrected", detail: "regenerated answer needed more support — escalating to synthesis" });
  } else if (stage === "verifying" && state.currentRecoveryPhase) {
    setStepState(timelineHandle, state.currentRecoveryPhase, { detail: "double-checking the new answer is grounded before returning it…" });
  }
}

function renderInstrumentation(instrumentation, finalStatus) {
  if (!timelineHandle.steps.verify.el.hidden) {
    if (instrumentation.initial_answer_valid) {
      setStepState(timelineHandle, "verify", {
        state: "done",
        detail: "passed verification on the first pass — grounded, complete, and on-target.",
      });
    } else {
      setStepState(timelineHandle, "verify", {
        state: "corrected",
        detail: "the fast answer needed a closer look, so the pipeline automatically escalated to re-ranking.",
      });
    }
  }

  if (instrumentation.rerank_triggered) {
    const rerankTime = instrumentation.rerank_latency_ms != null ? `${instrumentation.rerank_latency_ms.toFixed(0)} ms` : "";
    const sourceCount = instrumentation.selected_source_ids.length;
    if (!instrumentation.reranked_answer_generated || sourceCount === 0) {
      setStepState(timelineHandle, "rerank", {
        state: "corrected",
        hidden: false,
        detail: "none of the retrieved passages contained the answer — declined rather than guess.",
        time: rerankTime,
      });
    } else if (instrumentation.regenerated_answer_valid) {
      setStepState(timelineHandle, "rerank", {
        state: "done",
        hidden: false,
        detail: `selected ${sourceCount} source${sourceCount === 1 ? "" : "s"} from the full candidate pool and regenerated a verified answer.`,
        time: rerankTime,
      });
    } else {
      setStepState(timelineHandle, "rerank", {
        state: "corrected",
        hidden: false,
        detail: `selected ${sourceCount} source${sourceCount === 1 ? "" : "s"}, but the regenerated answer needed further verification.`,
        time: rerankTime,
      });
    }
  }

  if (instrumentation.synthesis_triggered) {
    if (finalStatus === "answered") {
      setStepState(timelineHandle, "synthesis", {
        state: "done",
        hidden: false,
        detail: "combined evidence across multiple passages to produce a verified answer.",
      });
    } else {
      setStepState(timelineHandle, "synthesis", {
        state: "corrected",
        hidden: false,
        detail: "no combination of retrieved passages fully supported an answer — declined rather than guess.",
      });
    }
  }
}

function renderResult(data) {
  resultViewEl.hidden = false;
  resultReasonEl.textContent = data.reason ?? "";

  // Badge morph fires together with the answer diff/refusal render below, not
  // before it -- the spec requires the badge to resolve in sync with the
  // diff, not pop in early while the old text is still showing.
  if (data.final_status === "answered" && data.answer) {
    showAnswer(data.answer.text, data.answer.source_passage_ids);
  } else {
    showRefusal(data.reason);
  }
  const meta = STATUS_META[data.final_status] ?? { label: data.final_status, cls: "warn" };
  outcomeBadgeEl.hidden = false;
  outcomeBadgeEl.textContent = meta.label;
  outcomeBadgeEl.className = `badge badge-${meta.cls}`;

  renderLatencyTable(data.pipeline);
  totalE2eLatencyMsEl.textContent = data.instrumentation.recovery_latency_ms.toFixed(2);

  renderInstrumentation(data.instrumentation, data.final_status);
}

const POLL_INTERVAL_MS = 700;

async function pollForFinalAnswer(requestId) {
  for (;;) {
    let data;
    try {
      data = await getAnswer(requestId);
    } catch (err) {
      showToast(err instanceof ApiError ? err.message : `polling for result failed: ${err.message}`);
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

function setAsking(isAsking) {
  for (const controls of controlSets) {
    controls.askBtn.disabled = isAsking;
    controls.askBtnSpinner.hidden = !isAsking;
    controls.askBtnLabel.textContent = isAsking ? "Asking…" : "Ask";
  }
}

async function runQuery(queryText) {
  if (!queryText.trim()) {
    // transcribeAndQuery() may have already set the spinner active before
    // calling us (e.g. a silent recording producing an empty transcript) --
    // clear it here so the Ask button doesn't stay stuck spinning forever.
    setAsking(false);
    return;
  }
  setAsking(true);
  resetResultView();

  const controls = activeControlSet();
  let data;
  try {
    data = await postAnswer({ queryText, language: controls.languageSelect.value, strategy: controls.strategySelect.value });
  } catch (err) {
    showToast(err instanceof ApiError ? err.message : `query failed: ${err.message}`);
    resultViewEl.hidden = true;
    setAsking(false);
    return;
  }

  if (!state.collapsed) {
    state.collapsed = true;
    // Carry the hero's actual language/strategy selection into the compact
    // bar's independent control set -- otherwise it silently reverts to
    // whatever loadCollections() defaulted it to (en/native) instead of
    // reflecting the query that was just run.
    compactControls.languageSelect.value = heroControls.languageSelect.value;
    refreshStrategiesFor(compactControls);
    compactControls.strategySelect.value = heroControls.strategySelect.value;
    state.activeControls = "compact";
    await collapseHero({ heroEl, compactBarEl });
  }

  renderFastPath(data);
  setAsking(false);
  await pollForFinalAnswer(data.request_id);
}

for (const controls of controlSets) {
  controls.askBtn.addEventListener("click", () => runQuery(controls.queryInput.value));
  controls.queryInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter") runQuery(controls.queryInput.value);
  });
}

async function transcribeAndQuery(blob, filename) {
  const controls = activeControlSet();
  // Reuses the Ask button's spinner state to cover the /api/transcribe
  // round trip -- runQuery() below sets it again once the transcript comes
  // back, which is a harmless no-op re-assertion of the same state.
  setAsking(true);
  let data;
  try {
    data = await transcribeAudio({ blob, filename, language: controls.languageSelect.value });
  } catch (err) {
    setAsking(false);
    showToast(err instanceof ApiError ? err.message : `transcription failed: ${err.message}`);
    return;
  }
  controls.queryInput.value = data.transcript;
  runQuery(data.transcript);
}

audioFileInputEl.addEventListener("change", () => {
  const file = audioFileInputEl.files[0];
  if (file) transcribeAndQuery(file, file.name);
});

const audioRecorder = createAudioRecorder({
  waveformEl,
  countdownEl: recordCountdownEl,
  onStop: (blob, filename) => {
    recordBtnEl.textContent = "Record";
    transcribeAndQuery(blob, filename);
  },
  onError: (err) => {
    recordBtnEl.textContent = "Record";
    showToast(`microphone access failed: ${err.message}`);
  },
});

recordBtnEl.addEventListener("click", () => {
  if (audioRecorder.isRecording()) {
    audioRecorder.stop();
  } else {
    recordBtnEl.textContent = "Stop";
    audioRecorder.start();
  }
});

loadCollections().catch((err) => showToast(`failed to load collections: ${err.message}`));
