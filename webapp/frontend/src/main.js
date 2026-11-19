import "./styles/tokens.css";
import "./styles/main.css";
import { fetchCollections, postAnswer, getAnswer, transcribeAudio, ApiError } from "./api.js";
import { initToastContainer, showToast, clearToasts } from "./toast.js";
import { initGradientMesh } from "./gradientMesh.js";
import { initTimeline, setStepState, STEP_NAMES } from "./timeline.js";
import { renderAnswerDiff } from "./answerDiff.js";
import { createAudioRecorder } from "./audioRecorder.js";
import { collapseHero } from "./heroChoreography.js";

// The recovery chain's validate/synthesize calls run on llm_model; the speculative
// drafting step runs a small drafter_model and scores drafts with verifier_model.
// The fast path never touches an LLM.
const STATUS_META = {
  answered: { label: "Verified Answer", cls: "good" },
  not_answered: { label: "No Direct Answer Found", cls: "warn" },
  insufficient_evidence: { label: "Insufficient Evidence", cls: "warn" },
  ungrounded: { label: "Answer Withheld — Not Grounded", cls: "warn" },
  fallback: { label: "No Verified Answer After Full Recovery", cls: "warn" },
  recovery_unavailable: { label: "Fast Answer — Verification Unavailable", cls: "warn" },
};

const state = {
  sarvamKeyConfigured: false,
  openaiKeyConfigured: false,
  llmModel: "LLM",
  drafterModel: "drafter",
  verifierModel: "verifier",
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
const retrievalCountEl = document.querySelector('[data-role="retrieval-count"]');
const retrievalListEl = document.querySelector('[data-role="retrieval-list"]');

const audioFileInputEl = heroEl.querySelector('[data-role="audio-file"]');
const heroRecordBtnEl = heroEl.querySelector('[data-role="record-btn"]');
const heroRecordLabelEl = heroEl.querySelector('[data-role="record-btn-label"]');
const waveformEl = heroEl.querySelector('[data-role="waveform"]');
const heroRecordCountdownEl = heroEl.querySelector('[data-role="record-countdown"]');
const audioDisabledNoteEl = heroEl.querySelector('[data-role="audio-disabled-note"]');
const transcriptPreviewEl = heroEl.querySelector('[data-role="transcript-preview"]');
const transcriptPreviewTextEl = heroEl.querySelector('[data-role="transcript-preview-text"]');
const compactRecordBtnEl = compactBarEl.querySelector('[data-role="record-btn"]');
const compactRecordCountdownEl = compactBarEl.querySelector('[data-role="record-countdown"]');

initToastContainer(toastContainerEl);
initGradientMesh(gradientMeshEl);
const timelineHandle = initTimeline(timelineRootEl);

function buildControlSet(containerEl) {
  const set = {
    languageSelect: containerEl.querySelector('[data-role="language-select"]'),
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

function strategyFor(language) {
  const strategies = strategiesByLanguage[language] ?? [];
  return strategies.includes("native") ? "native" : (strategies[0] ?? "native");
}

async function loadCollections() {
  const data = await fetchCollections();
  state.sarvamKeyConfigured = data.sarvam_key_configured;
  state.openaiKeyConfigured = data.openai_key_configured;
  state.llmModel = data.llm_model || "LLM";
  for (const tag of document.querySelectorAll('[data-role="llm-model"]')) tag.textContent = state.llmModel;
  state.drafterModel = data.drafter_model || state.drafterModel;
  state.verifierModel = data.verifier_model || state.verifierModel;
  for (const tag of document.querySelectorAll('[data-role="spec-models"]')) {
    tag.textContent = `${state.drafterModel} → ${state.verifierModel}`;
  }

  const languages = [...new Set(data.collections.map((c) => c.language))];
  strategiesByLanguage = {};
  for (const c of data.collections) {
    (strategiesByLanguage[c.language] ??= []).push(c.strategy);
  }

  for (const controls of controlSets) {
    populateSelect(controls.languageSelect, languages, "en");
  }

  if (!state.sarvamKeyConfigured) {
    audioFileInputEl.disabled = true;
    heroRecordBtnEl.disabled = true;
    compactRecordBtnEl.disabled = true;
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
  retrievalCountEl.textContent = "0 / 20";
  retrievalListEl.innerHTML = "";
  state.lastAnswerText = null;
  state.currentRecoveryPhase = null;
  for (const name of STEP_NAMES) {
    setStepState(timelineHandle, name, { state: "pending", detail: "", time: "" });
  }
  setStepState(timelineHandle, "rerank", { hidden: true });
  renderDrafts([]);
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
  answerTextEl.classList.remove("answer-text--streaming");
  renderAnswerDiff(answerTextEl, state.lastAnswerText, text);
  state.lastAnswerText = text;
  answerSourcesRowEl.hidden = false;
  answerSourceIdsEl.textContent = sourceIds.join(", ") || "none";
}

function showStreamingAnswer(text) {
  answerSkeletonEl.hidden = true;
  refusalTextEl.hidden = true;
  answerTextEl.hidden = false;
  answerTextEl.classList.add("answer-text--streaming");
  answerTextEl.textContent = text;
  state.lastAnswerText = text;
  answerSourcesRowEl.hidden = true;
}

function showRefusal(reason) {
  answerSkeletonEl.hidden = true;
  answerTextEl.hidden = true;
  answerTextEl.classList.remove("answer-text--streaming");
  answerSourcesRowEl.hidden = true;
  refusalTextEl.hidden = false;
  refusalTextEl.textContent = reason || "No validated answer could be produced.";
  state.lastAnswerText = null;
}

function willRunVerify(pipelineOutcome) {
  return pipelineOutcome === "answered";
}

function renderRetrievedCandidates(candidates = [], selectedSourceIds = []) {
  retrievalListEl.innerHTML = "";
  const visible = candidates.slice(0, 20);
  retrievalCountEl.textContent = `${visible.length} / 20`;

  if (!visible.length) {
    const empty = document.createElement("p");
    empty.className = "retrieval-empty";
    empty.textContent = "No passages were returned.";
    retrievalListEl.appendChild(empty);
    return;
  }

  const selected = new Set(selectedSourceIds);
  for (const candidate of visible) {
    const card = document.createElement("article");
    card.className = `retrieval-item${selected.has(candidate.source_passage_id) ? " retrieval-item--selected" : ""}`;

    const header = document.createElement("div");
    header.className = "retrieval-item-header";
    const rank = document.createElement("span");
    rank.className = "retrieval-rank";
    rank.textContent = `0${candidate.rank}`.slice(-2);
    const score = document.createElement("span");
    score.className = "retrieval-score";
    score.textContent = `score ${Number(candidate.score).toFixed(2)}`;
    header.append(rank, score);

    const text = document.createElement("p");
    text.className = "retrieval-text";
    text.dir = "auto";
    text.textContent = candidate.text;

    const meta = document.createElement("p");
    meta.className = "retrieval-meta";
    meta.textContent = `${candidate.strategy} · ${candidate.query_type} · ${candidate.source_passage_id}`;

    card.append(header, text, meta);
    retrievalListEl.appendChild(card);
  }
}

function renderFastPath(data) {
  resultViewEl.hidden = false;
  renderRetrievedCandidates(data.pipeline.retrieved_candidates);

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
      detail: `${state.llmModel} is checking the answer is grounded, complete, and on-target…`,
    });
  } else {
    setStepState(timelineHandle, "verify", { hidden: true });
    setStepState(timelineHandle, "rerank", {
      state: "active",
      hidden: false,
      detail: "grouping the retrieved passages by topic before drafting…",
    });
  }
}

function renderStageProgress(stage) {
  if (stage === "drafting") {
    state.currentRecoveryPhase = "rerank";
    setStepState(timelineHandle, "rerank", {
      state: "active",
      hidden: false,
      detail: `${state.drafterModel} is writing several drafts at once, each from a different mix of evidence…`,
    });
    if (!timelineHandle.steps.verify.el.hidden) {
      setStepState(timelineHandle, "verify", { state: "corrected", detail: "flagged for a closer look — escalating to speculative drafting" });
    }
  } else if (stage === "verifying_drafts") {
    setStepState(timelineHandle, "rerank", {
      detail: `${state.verifierModel} is scoring each draft against its own rationale…`,
    });
  } else if (stage === "synthesizing") {
    state.currentRecoveryPhase = "synthesis";
    setStepState(timelineHandle, "synthesis", {
      state: "active",
      hidden: false,
      detail: `${state.llmModel} is synthesizing an answer across multiple passages…`,
    });
    setStepState(timelineHandle, "rerank", { state: "corrected", detail: "no draft held up — escalating to synthesis across every passage" });
  } else if (stage === "verifying" && state.currentRecoveryPhase) {
    setStepState(timelineHandle, state.currentRecoveryPhase, { detail: "double-checking the new answer is grounded before returning it…" });
  }
}

// One line per draft, best first: score, then the answer it proposed. The
// score is draft confidence x verifier support, shown so a reader can see
// why the winner won, e.g. "0.87  Islamabad is the capital of Pakistan."
function renderDrafts(drafts) {
  const list = timelineHandle.steps.rerank.el.querySelector('[data-role="draft-list"]');
  list.innerHTML = "";
  list.hidden = drafts.length === 0;
  drafts.forEach((draft, i) => {
    const item = document.createElement("li");
    item.className = `draft-item${i === 0 ? " draft-item--best" : ""}`;
    const score = document.createElement("span");
    score.className = "draft-score";
    score.textContent = draft.score.toFixed(2);
    score.title = `draft confidence ${draft.draft_confidence.toFixed(2)} × verifier support ${draft.support.toFixed(2)}`;
    const answer = document.createElement("span");
    answer.className = "draft-answer";
    answer.dir = "auto";
    answer.textContent = draft.answer;
    item.append(score, answer);
    list.appendChild(item);
  });
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
        detail: "the fast answer needed a closer look, so the pipeline escalated to speculative drafting.",
      });
    }
  }

  if (instrumentation.rerank_triggered) {
    const rerankTime = instrumentation.rerank_latency_ms != null ? `${instrumentation.rerank_latency_ms.toFixed(0)} ms` : "";
    const drafts = instrumentation.drafts ?? [];
    const best = drafts[0];
    renderDrafts(drafts);
    if (!best) {
      setStepState(timelineHandle, "rerank", {
        state: "corrected",
        hidden: false,
        detail: "every draft said its evidence didn't answer the question — declined rather than guess.",
        time: rerankTime,
      });
    } else if (instrumentation.regenerated_answer_valid) {
      setStepState(timelineHandle, "rerank", {
        state: "done",
        hidden: false,
        detail: `${drafts.length} draft${drafts.length === 1 ? "" : "s"} survived; the verifier picked the best (score ${best.score.toFixed(2)}) and it passed the grounding check.`,
        time: rerankTime,
      });
    } else {
      setStepState(timelineHandle, "rerank", {
        state: "corrected",
        hidden: false,
        detail: `the top draft (score ${best.score.toFixed(2)}) didn't pass the final grounding check.`,
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
  renderRetrievedCandidates(
    data.pipeline.retrieved_candidates,
    data.answer?.source_passage_ids ?? [],
  );

  // Badge morph fires together with the answer diff/refusal render below, not
  // before it -- the spec requires the badge to resolve in sync with the
  // diff, not pop in early while the old text is still showing.
  if (data.answer) {
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

const POLL_INTERVAL_MS = 250;

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
    if (data.partial_answer) showStreamingAnswer(data.partial_answer);
    renderStageProgress(data.stage);
    await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
  }
}

function setAsking(isAsking) {
  for (const controls of controlSets) {
    controls.askBtn.disabled = isAsking;
    controls.askBtnSpinner.hidden = !isAsking;
    controls.askBtnLabel.textContent = isAsking ? "Asking…" : "Ask →";
  }
}

function detectScriptLanguage(text) {
  if (/[ऀ-ॿ]/.test(text)) return "hi";
  if (/[஀-௿]/.test(text)) return "ta";
  if (/[؀-ۿ]/.test(text)) return "ur";
  return null; // Latin or unknown: keep whatever the dropdown says
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
  // The dropdown defaults to English, and BM25 over the English index finds
  // nothing for a Hindi question. So pick the language from the script the
  // user actually typed: "ताजमहल कहाँ है" -> hi, Arabic script -> ur, Tamil -> ta.
  const scriptLanguage = detectScriptLanguage(queryText);
  if (scriptLanguage && [...controls.languageSelect.options].some((o) => o.value === scriptLanguage)) {
    controls.languageSelect.value = scriptLanguage;
  }
  let data;
  try {
    data = await postAnswer({ queryText, language: controls.languageSelect.value, strategy: strategyFor(controls.languageSelect.value) });
  } catch (err) {
    showToast(err instanceof ApiError ? err.message : `query failed: ${err.message}`);
    resultViewEl.hidden = true;
    setAsking(false);
    return;
  }

  if (!state.collapsed) {
    state.collapsed = true;
    // Carry the hero's actual language selection into the compact bar's
    // independent control set -- otherwise it silently reverts to whatever
    // loadCollections() defaulted it to (en) instead of reflecting the
    // query that was just run.
    compactControls.languageSelect.value = heroControls.languageSelect.value;
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
  if (transcriptPreviewEl && transcriptPreviewTextEl) {
    transcriptPreviewTextEl.textContent = data.transcript;
    transcriptPreviewEl.hidden = false;
  }
  runQuery(data.transcript);
}

audioFileInputEl.addEventListener("change", () => {
  const file = audioFileInputEl.files[0];
  if (file) transcribeAndQuery(file, file.name);
});

function wireRecorder({ recordBtnEl, waveformEl, countdownEl, labelEl }) {
  function reset() {
    recordBtnEl.classList.remove("recording");
    recordBtnEl.setAttribute("aria-label", "Record a question");
    if (labelEl) labelEl.textContent = "Start Recording";
  }

  const recorder = createAudioRecorder({
    waveformEl,
    countdownEl,
    onStop: (blob, filename) => {
      reset();
      transcribeAndQuery(blob, filename);
    },
    onError: (err) => {
      reset();
      showToast(`microphone access failed: ${err.message}`);
    },
  });

  recordBtnEl.addEventListener("click", () => {
    if (recorder.isRecording()) {
      recorder.stop();
    } else {
      recordBtnEl.classList.add("recording");
      recordBtnEl.setAttribute("aria-label", "Stop recording");
      if (labelEl) labelEl.textContent = "Stop Recording";
      if (transcriptPreviewEl) transcriptPreviewEl.hidden = true;
      recorder.start();
    }
  });

  return recorder;
}

wireRecorder({ recordBtnEl: heroRecordBtnEl, waveformEl, countdownEl: heroRecordCountdownEl, labelEl: heroRecordLabelEl });
wireRecorder({ recordBtnEl: compactRecordBtnEl, waveformEl: null, countdownEl: compactRecordCountdownEl });

loadCollections().catch((err) => showToast(`failed to load collections: ${err.message}`));
