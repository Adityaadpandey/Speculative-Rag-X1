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
const answerBlock = document.getElementById("answer-block");
const answerText = document.getElementById("answer-text");
const answerSourceId = document.getElementById("answer-source-id");
const answerRetrievalScore = document.getElementById("answer-retrieval-score");
const answerGroundednessScore = document.getElementById("answer-groundedness-score");
const refusalText = document.getElementById("refusal-text");
const latencyTableBody = document.querySelector("#latency-table tbody");
const reviewPanel = document.getElementById("review-panel");
const reviewStatus = document.getElementById("review-status");
const reviewResult = document.getElementById("review-result");
const reviewBadge = document.getElementById("review-badge");
const reviewReasoning = document.getElementById("review-reasoning");
const reviewLatencyMs = document.getElementById("review-latency-ms");
const rerankPanel = document.getElementById("rerank-panel");
const rerankStatus = document.getElementById("rerank-status");
const rerankResult = document.getElementById("rerank-result");
const rerankText = document.getElementById("rerank-text");
const rerankSourceId = document.getElementById("rerank-source-id");
const rerankRetrievalScore = document.getElementById("rerank-retrieval-score");
const rerankLatencyMs = document.getElementById("rerank-latency-ms");
const rerankReasoning = document.getElementById("rerank-reasoning");
const reviewFinalStatus = document.getElementById("review-final-status");
const reviewFinalReason = document.getElementById("review-final-reason");
const synthesisPanel = document.getElementById("synthesis-panel");
const synthesisStatus = document.getElementById("synthesis-status");
const synthesisResult = document.getElementById("synthesis-result");
const synthesisAnswer = document.getElementById("synthesis-answer");
const synthesisAnswerable = document.getElementById("synthesis-answerable");
const synthesisSources = document.getElementById("synthesis-sources");
const synthesisLatencyMs = document.getElementById("synthesis-latency-ms");
const synthesisReasoning = document.getElementById("synthesis-reasoning");
const errorBanner = document.getElementById("error-banner");

const OUTCOME_LABELS = {
  answered: "ANSWERED",
  off_topic_refused: "OFF-TOPIC",
  unsafe_input_refused: "UNSAFE INPUT",
  ungrounded_refused: "UNGROUNDED",
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
  languageSelect.value = languages.includes("hi") ? "hi" : (languages[0] ?? "");

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
  answerBlock.hidden = true;
  refusalText.hidden = true;
  reviewResult.hidden = true;
  reviewStatus.hidden = false;
  rerankResult.hidden = true;
  rerankStatus.hidden = false;
  synthesisPanel.hidden = true;
  synthesisResult.hidden = true;
  synthesisStatus.hidden = false;
  latencyTableBody.innerHTML = "";
}

function renderResult(data) {
  resultPanel.hidden = false;
  outcomeBadge.textContent = OUTCOME_LABELS[data.outcome] ?? data.outcome;
  outcomeBadge.className = `badge badge-${data.outcome}`;

  if (data.outcome === "answered") {
    answerBlock.hidden = false;
    refusalText.hidden = true;
    answerText.textContent = data.answer.text;
    answerSourceId.textContent = data.answer.source_passage_id;
    answerRetrievalScore.textContent = data.answer.retrieval_score.toFixed(3);
    answerGroundednessScore.textContent = data.answer.groundedness_score.toFixed(3);
  } else {
    answerBlock.hidden = true;
    refusalText.hidden = false;
    refusalText.textContent = data.refusal_reason;
  }

  latencyTableBody.innerHTML = "";
  for (const timing of data.timings) {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${timing.stage}</td><td>${timing.elapsed_ms.toFixed(2)} ms</td>`;
    latencyTableBody.appendChild(row);
  }
  const totalRow = document.createElement("tr");
  totalRow.innerHTML = `<td><strong>TOTAL</strong></td><td><strong>${data.total_ms.toFixed(2)} ms</strong></td>`;
  latencyTableBody.appendChild(totalRow);
}

async function startReview(queryText, answerTextValue) {
  reviewPanel.hidden = false;
  reviewStatus.hidden = false;
  reviewResult.hidden = true;

  if (!state.anthropicKeyConfigured) {
    reviewStatus.textContent = "ANTHROPIC_API_KEY not set — skipping groundedness review";
    return;
  }

  reviewStatus.textContent = "checking groundedness...";
  try {
    const response = await fetch("/api/reflect", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query_text: queryText,
        language: languageSelect.value,
        strategy: strategySelect.value,
        answer_text: answerTextValue,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      reviewStatus.textContent = `review failed: ${data.error ?? "unknown error"}`;
      return;
    }
    reviewStatus.hidden = true;
    reviewResult.hidden = false;
    reviewBadge.textContent = data.grounded ? "GROUNDED" : "NOT GROUNDED";
    reviewBadge.className = `badge ${data.grounded ? "badge-grounded" : "badge-not-grounded"}`;
    reviewReasoning.textContent = data.reasoning;
    reviewLatencyMs.textContent = data.reflect_latency_ms.toFixed(2);
    reviewFinalStatus.textContent = data.final_status ?? "pending";
    reviewFinalReason.textContent = data.reason ?? "";
    if (data.final_status && data.final_status !== "answered") {
      startSynthesis(queryText);
    }
  } catch (err) {
    reviewStatus.textContent = `review failed: ${err.message}`;
  }
}

async function startRerank(queryText) {
  rerankPanel.hidden = false;
  rerankStatus.hidden = false;
  rerankResult.hidden = true;

  if (!state.anthropicKeyConfigured) {
    rerankStatus.textContent = "ANTHROPIC_API_KEY not set — skipping rerank check";
    return;
  }

  rerankStatus.textContent = "reranking candidates...";
  try {
    const response = await fetch("/api/rerank", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query_text: queryText,
        language: languageSelect.value,
        strategy: strategySelect.value,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      rerankStatus.textContent = `rerank failed: ${data.error ?? "unknown error"}`;
      return;
    }
    rerankStatus.hidden = true;
    rerankResult.hidden = false;
    rerankText.textContent = data.text;
    rerankSourceId.textContent = data.source_passage_id;
    rerankRetrievalScore.textContent = data.retrieval_score.toFixed(3);
    rerankLatencyMs.textContent = data.rerank_latency_ms.toFixed(2);
    rerankReasoning.textContent = data.reasoning;
  } catch (err) {
    rerankStatus.textContent = `rerank failed: ${err.message}`;
  }
}

async function startSynthesis(queryText) {
  synthesisPanel.hidden = false;
  synthesisStatus.hidden = false;
  synthesisResult.hidden = true;

  if (!state.anthropicKeyConfigured) {
    synthesisStatus.textContent = "ANTHROPIC_API_KEY not set — skipping synthesis";
    return;
  }

  synthesisStatus.textContent = "checking multiple passages...";
  try {
    const response = await fetch("/api/synthesize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        query_text: queryText,
        language: languageSelect.value,
        strategy: strategySelect.value,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      synthesisStatus.textContent = `synthesis failed: ${data.error ?? "unknown error"}`;
      return;
    }
    synthesisStatus.hidden = true;
    synthesisResult.hidden = false;
    synthesisAnswerable.textContent = data.answerable;
    synthesisAnswer.textContent = data.answerable
      ? data.answer_text
      : "The retrieved evidence does not contain enough information to answer this fully.";
    synthesisSources.textContent = data.sources.map((s) => s.source_passage_id).join(", ") || "none";
    synthesisLatencyMs.textContent = data.synthesis_latency_ms.toFixed(2);
    synthesisReasoning.textContent = data.reasoning;
  } catch (err) {
    synthesisStatus.textContent = `synthesis failed: ${err.message}`;
  }
}

async function runQuery(queryText) {
  resetPanels();
  if (!queryText.trim()) {
    return;
  }
  let response;
  let data;
  try {
    response = await fetch("/api/query", {
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
    return;
  }
  if (!response.ok) {
    showError(data.error ?? "query failed");
    return;
  }

  renderResult(data);
  if (data.outcome === "answered") {
    startReview(queryText, data.answer.text);
    startRerank(queryText);
  } else if (data.outcome === "insufficient_evidence") {
    startSynthesis(queryText);
  }
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
