import {
  SUPPORTED_MODELS,
  VERIFIED_SAMPLES,
  CONTEXT_EXPERIMENTS,
  RACE_PROMPTS,
  SANDBOX_PRESETS,
  QUIZ_DATA,
  CONCEPTS_DATA
} from "./data.js";

/* ============================================================================
   1. DOM REFERENCES & STORAGE HELPERS
   ============================================================================ */
const protocolBanner = document.getElementById("protocol-banner");
const themeToggleBtn = document.getElementById("theme-toggle");
const themeIcon = document.getElementById("theme-icon");
const viewTabs = document.querySelectorAll(".view-tab");
const viewPanels = document.querySelectorAll(".view-panel");

const modelSelect = document.getElementById("model-select");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const cacheBadge = document.getElementById("cache-badge");
const analyzerEyebrow = document.getElementById("analyzer-eyebrow");
const modelCallout = document.getElementById("model-callout");
const calloutText = document.getElementById("callout-text");
const calloutPct = document.getElementById("callout-pct");
const loadProgressFill = document.getElementById("load-progress-fill");
const errorBox = document.getElementById("error-box");

const sentenceInput = document.getElementById("sentence-input");
const sentenceInputB = document.getElementById("sentence-input-b");
const autoAnalyzeToggle = document.getElementById("auto-analyze-toggle");
const tempSlider = document.getElementById("temp-slider");
const topkSlider = document.getElementById("topk-slider");
const toppSlider = document.getElementById("topp-slider");
const btnToggleCompare = document.getElementById("btn-toggle-compare");
const compareBox = document.getElementById("compare-box");
const compareResult = document.getElementById("compare-result");
const expRow = document.getElementById("context-exp-row");
const btnShareLink = document.getElementById("btn-share-link");

const visualSlider = document.getElementById("visual-slider");
const curvePath = document.getElementById("ppl-curve-path");
const curveDot = document.getElementById("ppl-curve-dot");

const xrayInput = document.getElementById("xray-input");
const xrayTokens = document.getElementById("xray-tokens");
const xrayStats = document.getElementById("xray-stats");
const xrayBadge = document.getElementById("xray-mode-badge");

const qTextEl = document.getElementById("quiz-question");
const qOptsEl = document.getElementById("quiz-options");
const qFeedbackEl = document.getElementById("quiz-feedback");
const qProgressText = document.getElementById("quiz-progress-text");
const qProgressFill = document.getElementById("quiz-progress-fill");
const qScoreEl = document.getElementById("quiz-score");
const qStreakEl = document.getElementById("quiz-streak");
const btnNextQ = document.getElementById("btn-next-question");

const racePresetsEl = document.getElementById("race-presets");
const racePrefixInput = document.getElementById("race-prefix-input");
const raceContextDisplay = document.getElementById("race-context-display");
const raceGuessInput = document.getElementById("race-guess-input");
const raceRoundBadge = document.getElementById("race-round-badge");
const raceHumanPplEl = document.getElementById("race-human-ppl");
const raceAiPplEl = document.getElementById("race-ai-ppl");
const raceFeedbackEl = document.getElementById("race-round-feedback");
const raceHistoryEl = document.getElementById("race-history-list");

function readStorage(key, fallback) {
  try {
    return localStorage.getItem(key) || fallback;
  } catch {
    return fallback;
  }
}

function writeStorage(key, val) {
  try {
    localStorage.setItem(key, val);
  } catch {
    // Ignore storage quota or sandbox restrictions
  }
}

/* ============================================================================
   2. THEME & NAVIGATION
   ============================================================================ */
if (window.location.protocol === "file:") {
  protocolBanner.hidden = false;
}

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  themeIcon.textContent = theme === "dark" ? "☀" : "☾";
  writeStorage("perplexity-lab-theme", theme);
}

applyTheme(readStorage("perplexity-lab-theme", "light"));
themeToggleBtn.addEventListener("click", () => {
  const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
  applyTheme(next);
});

viewTabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    const target = tab.dataset.view;
    viewTabs.forEach((t) => {
      const isMatch = t === tab;
      t.classList.toggle("active", isMatch);
      t.setAttribute("aria-selected", String(isMatch));
    });
    viewPanels.forEach((panel) => {
      const isMatch = panel.id === `view-${target}`;
      panel.classList.toggle("active", isMatch);
      panel.hidden = !isMatch;
    });
  });
});

/* ============================================================================
   3. MULTI-MODEL ENGINE (WITH SHARED IN-FLIGHT PROMISE) & CACHE INDICATOR
   ============================================================================ */
let transformersMod = null;
const loadedEngines = {};
const loadingPromises = {};
let activeModelId = "Xenova/distilgpt2";

let currentAnalysis = null;
let selectedTokenIdx = 0;
let currentTemperature = 1.0;
let currentTopK = 8;
let currentTopP = 1.0;

function isActiveModelReady() {
  return Boolean(loadedEngines[activeModelId]);
}

function getActiveEngine() {
  return loadedEngines[activeModelId] || null;
}

function setEngineStatus(state, label) {
  statusDot.className = `status-dot ${state}`;
  statusText.textContent = label;
}

function setProgress(pct, text) {
  const clamped = Math.max(0, Math.min(100, Math.round(pct)));
  loadProgressFill.style.width = `${clamped}%`;
  calloutPct.textContent = `${clamped}%`;
  if (text) calloutText.textContent = text;
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.hidden = false;
}

function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

async function refreshCacheBadge() {
  if (!("caches" in window)) {
    cacheBadge.hidden = true;
    return;
  }
  try {
    const cache = await caches.open("transformers-cache");
    const keys = await cache.keys();
    const slug = activeModelId.split("/")[1];
    const hasWeights = keys.some((req) => req.url.includes(slug));
    cacheBadge.hidden = !hasWeights;
  } catch {
    cacheBadge.hidden = true;
  }
}

function updateModelHeaderUI() {
  const meta = SUPPORTED_MODELS[activeModelId];
  analyzerEyebrow.textContent = `Real-Time Causal Inference · ${meta.label} (${meta.vocabSize.toLocaleString()} shared BPE vocab)`;
  if (isActiveModelReady()) {
    setEngineStatus("ready", `${meta.short} Ready`);
  } else if (activeModelId === "Xenova/distilgpt2") {
    setEngineStatus("", "Preview Mode");
  } else {
    setEngineStatus("", `${meta.short} Not Loaded`);
  }
  refreshCacheBadge();
  updateXRay();
}

modelSelect.addEventListener("change", () => {
  activeModelId = modelSelect.value;
  updateModelHeaderUI();
});

async function ensureModelLoaded(targetModelId = activeModelId) {
  if (loadedEngines[targetModelId]) return true;
  if (loadingPromises[targetModelId]) return loadingPromises[targetModelId];

  const meta = SUPPORTED_MODELS[targetModelId];
  loadingPromises[targetModelId] = (async () => {
    setEngineStatus("loading", `Loading ${meta.short}...`);
    modelCallout.hidden = false;
    setProgress(8, `Loading Transformers.js runtime for ${meta.short}...`);

    try {
      if (!transformersMod) {
        transformersMod = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3");
      }
      const { AutoTokenizer, AutoModelForCausalLM } = transformersMod;

      setProgress(18, `Downloading ${meta.short} BPE tokenizer...`);
      const tokenizer = await AutoTokenizer.from_pretrained(targetModelId, {
        progress_callback: (p) => {
          if (p.status === "progress" && typeof p.progress === "number") {
            setProgress(18 + p.progress * 0.22);
          }
        }
      });

      setProgress(42, `Downloading ${meta.short} quantized ONNX weights...`);
      const model = await AutoModelForCausalLM.from_pretrained(targetModelId, {
        dtype: "q8",
        progress_callback: (p) => {
          if (p.status === "progress" && typeof p.progress === "number") {
            setProgress(42 + p.progress * 0.56);
          }
        }
      });

      loadedEngines[targetModelId] = { tokenizer, model };
      setProgress(100, `${meta.short} ready in browser.`);
      updateModelHeaderUI();
      setTimeout(() => {
        modelCallout.hidden = true;
      }, 2200);
      return true;
    } catch (err) {
      console.error("Model load error:", err);
      setEngineStatus("error", "Model Load Failed");
      modelCallout.hidden = true;
      showError(`Could not load ${meta.short} (${err.message}).`);
      return false;
    } finally {
      delete loadingPromises[targetModelId];
    }
  })();

  return loadingPromises[targetModelId];
}

/* ============================================================================
   4. MATH HELPERS & CAUSAL INFERENCE
   ============================================================================ */
function stableSoftmax(logits, temperature = 1.0) {
  const invT = 1 / Math.max(0.05, temperature);
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) {
    const scaled = logits[i] * invT;
    if (scaled > max) max = scaled;
  }
  const probs = new Float32Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] * invT - max);
    probs[i] = e;
    sum += e;
  }
  for (let i = 0; i < logits.length; i++) {
    probs[i] /= sum;
  }
  return probs;
}

function formatDisplayToken(text) {
  return text.replace(/ /g, "␣").replace(/\n/g, "↵");
}

function escapeHtml(str) {
  return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function extractTopKFromLogits(logits, tokenizer, temperature, k = 15) {
  const probs = stableSoftmax(logits, temperature);
  const top = [];
  for (let i = 0; i < probs.length; i++) {
    const p = probs[i];
    if (top.length < k || p > top[top.length - 1].probability) {
      top.push({ tokenId: i, probability: p });
      top.sort((a, b) => b.probability - a.probability);
      if (top.length > k) top.pop();
    }
  }
  return top.map((item, idx) => {
    const text = tokenizer.decode([item.tokenId]);
    return {
      rank: idx + 1,
      tokenId: item.tokenId,
      text,
      displaySymbol: formatDisplayToken(text),
      probability: item.probability
    };
  });
}

async function forwardCausalLM(engine, rawIds, targetModelId) {
  const { Tensor } = transformersMod;
  const bosId = 50256;
  const inputIds = [bosId, ...rawIds];
  if (inputIds.length > 1024) {
    throw new Error(`Input exceeds GPT-2's 1024-token context limit (${inputIds.length} tokens).`);
  }

  const tensorInput = new Tensor("int64", BigInt64Array.from(inputIds.map(BigInt)), [1, inputIds.length]);
  const tensorMask = new Tensor("int64", BigInt64Array.from(inputIds.map(() => 1n)), [1, inputIds.length]);
  const out = await engine.model({ input_ids: tensorInput, attention_mask: tensorMask });
  if (!out || !out.logits || !out.logits.data) {
    throw new Error("Model forward pass returned empty logits.");
  }

  const vocabSize = out.logits.dims?.[2] ?? SUPPORTED_MODELS[targetModelId].vocabSize;
  return { logitsData: out.logits.data, vocabSize };
}

async function runLiveInference(sentenceText, targetModelId = activeModelId) {
  const engine = loadedEngines[targetModelId];
  const { tokenizer } = engine;

  const encoded = tokenizer(sentenceText);
  const rawIds = Array.from(encoded.input_ids.data).map(Number);
  if (rawIds.length === 0) throw new Error("Input produced zero tokens.");

  const { logitsData, vocabSize } = await forwardCausalLM(engine, rawIds, targetModelId);
  const tokens = [];
  let sumLogProbs = 0;

  for (let i = 0; i < rawIds.length; i++) {
    const targetId = rawIds[i];
    const decodedText = tokenizer.decode([targetId]);
    const rowSlice = logitsData.subarray(i * vocabSize, (i + 1) * vocabSize);
    const probs = stableSoftmax(rowSlice, 1.0);
    const actualProb = Math.max(1e-12, probs[targetId]);
    sumLogProbs += Math.log(actualProb);

    const ctxTokens = ["[BOS]", ...rawIds.slice(0, i).map((id) => formatDisplayToken(tokenizer.decode([id])))];
    tokens.push({
      index: i + 1,
      text: decodedText,
      displaySymbol: formatDisplayToken(decodedText),
      tokenId: targetId,
      probability: actualProb,
      inverseProbability: 1 / actualProb,
      context: ctxTokens.join(" ")
    });
  }

  const lastRowOffset = rawIds.length * vocabSize;
  const lastLogits = new Float32Array(logitsData.subarray(lastRowOffset, lastRowOffset + vocabSize));
  const topCandidates = extractTopKFromLogits(lastLogits, tokenizer, 1.0, 15);

  const crossEntropy = -sumLogProbs / rawIds.length;
  const perplexity = Math.exp(crossEntropy);

  return {
    modelId: targetModelId,
    sentence: sentenceText,
    tokens,
    perplexity,
    crossEntropy,
    top8Next: topCandidates,
    rawLastLogits: lastLogits,
    isLive: true
  };
}

async function evaluateSequence(text, allowPreviewFallback = true) {
  const clean = text.trim();
  if (!clean) throw new Error("Please enter a non-empty sentence.");

  if (!isActiveModelReady() && allowPreviewFallback && activeModelId === "Xenova/distilgpt2" && VERIFIED_SAMPLES[clean]) {
    return { ...VERIFIED_SAMPLES[clean], modelId: "Xenova/distilgpt2", sentence: clean, isLive: false };
  }

  const loaded = await ensureModelLoaded(activeModelId);
  if (!loaded) {
    if (activeModelId === "Xenova/distilgpt2" && VERIFIED_SAMPLES[clean]) {
      return { ...VERIFIED_SAMPLES[clean], modelId: "Xenova/distilgpt2", sentence: clean, isLive: false };
    }
    throw new Error("Live model could not be loaded.");
  }

  return await runLiveInference(clean, activeModelId);
}

async function analyzeSentence(sentenceText) {
  const clean = sentenceText.trim();
  if (!clean) {
    showError("Please enter a sentence to evaluate.");
    return;
  }
  clearError();

  if (!isActiveModelReady() && activeModelId === "Xenova/distilgpt2" && VERIFIED_SAMPLES[clean]) {
    renderAnalysis({ ...VERIFIED_SAMPLES[clean], modelId: "Xenova/distilgpt2", sentence: clean, isLive: false });
    return;
  }

  setEngineStatus("analyzing", `Analyzing (${SUPPORTED_MODELS[activeModelId].short})...`);
  try {
    const result = await evaluateSequence(clean, false);
    renderAnalysis(result);
    updateModelHeaderUI();
  } catch (err) {
    if (!isActiveModelReady()) setEngineStatus("error", "Model Unavailable");
    showError(err.message);
  }
}

/* ============================================================================
   5. RENDER LIVE ANALYZER WORKBENCH + SURPRISAL SPARKLINE + SAMPLING
   ============================================================================ */
function surpriseLevel(prob) {
  const bits = -Math.log2(Math.max(1e-12, prob));
  if (bits <= 5) return "low";
  if (bits <= 10) return "mid";
  return "high";
}

function renderAnalysis(data) {
  currentAnalysis = data;
  selectedTokenIdx = Math.min(selectedTokenIdx, data.tokens.length - 1);
  if (selectedTokenIdx < 0) selectedTokenIdx = 0;

  renderHeatmapAndTable(data.tokens);
  renderSurprisalChart(data.tokens);
  selectToken(selectedTokenIdx);
  renderGaugeAndMath(data);
  renderNextPredictions();
}

function renderHeatmapAndTable(tokens) {
  const heatmap = document.getElementById("tokens-heatmap");
  const tbody = document.getElementById("tokens-table-body");
  document.getElementById("token-count-badge").textContent = `${tokens.length} token${tokens.length === 1 ? "" : "s"}`;

  heatmap.innerHTML = "";
  tbody.innerHTML = "";

  tokens.forEach((t, idx) => {
    const level = surpriseLevel(t.probability);
    const pct = (t.probability * 100).toFixed(2);
    const bits = (-Math.log2(Math.max(1e-12, t.probability))).toFixed(2);

    const chip = document.createElement("button");
    chip.type = "button";
    chip.className = `hm-token surprise-${level}`;
    chip.dataset.idx = String(idx);
    chip.innerHTML = `
      <span class="hm-text">${escapeHtml(t.displaySymbol)}</span>
      <span class="hm-meta">${pct}% · ${bits}b</span>
    `;
    chip.addEventListener("click", () => selectToken(idx));
    heatmap.appendChild(chip);

    const tr = document.createElement("tr");
    tr.dataset.idx = String(idx);
    tr.innerHTML = `
      <td class="mono muted-xs">${t.index}</td>
      <td class="mono"><strong>${escapeHtml(t.displaySymbol)}</strong></td>
      <td class="mono muted-xs">${t.tokenId}</td>
      <td class="mono">${pct}%</td>
      <td class="mono">${t.inverseProbability.toFixed(1)}</td>
      <td class="mono">${bits}</td>
      <td>
        <div class="mini-bar-track">
          <div class="mini-bar-fill ${level === "low" ? "" : level}" style="width: ${Math.min(100, Math.max(3, Number(pct)))}%;"></div>
        </div>
      </td>
    `;
    tr.addEventListener("click", () => selectToken(idx));
    tbody.appendChild(tr);
  });
}

function renderSurprisalChart(tokens) {
  const svg = document.getElementById("surprisal-svg");
  const peakNote = document.getElementById("peak-surprise-note");
  if (!tokens.length) {
    svg.innerHTML = "";
    peakNote.textContent = "Peak: —";
    return;
  }

  const W = 640;
  const H = 120;
  const padL = 34;
  const padR = 18;
  const padT = 14;
  const padB = 22;
  const plotW = W - padL - padR;
  const plotH = H - padT - padB;

  const bitsArr = tokens.map((t) => -Math.log2(Math.max(1e-12, t.probability)));
  const maxBits = Math.max(12, Math.ceil(Math.max(...bitsArr) / 2) * 2);

  let peakIdx = 0;
  bitsArr.forEach((b, i) => { if (b > bitsArr[peakIdx]) peakIdx = i; });
  peakNote.textContent = `Peak: ${tokens[peakIdx].displaySymbol} (${bitsArr[peakIdx].toFixed(2)} bits)`;

  const coords = tokens.map((t, i) => {
    const x = tokens.length === 1 ? padL + plotW / 2 : padL + (i / (tokens.length - 1)) * plotW;
    const y = padT + (1 - bitsArr[i] / maxBits) * plotH;
    return { x, y, bits: bitsArr[i], token: t, idx: i };
  });

  const gridLines = [0, 5, 10, maxBits].filter((v, idx, arr) => v <= maxBits && arr.indexOf(v) === idx)
    .map((val) => {
      const y = padT + (1 - val / maxBits) * plotH;
      return `
        <line x1="${padL}" y1="${y.toFixed(1)}" x2="${W - padR}" y2="${y.toFixed(1)}" stroke="var(--border)" stroke-dasharray="3 3" stroke-width="1"/>
        <text x="${padL - 6}" y="${(y + 3).toFixed(1)}" text-anchor="end" class="svg-label">${val}b</text>
      `;
    }).join("");

  const polyPts = coords.map((c) => `${c.x.toFixed(1)},${c.y.toFixed(1)}`).join(" ");
  const areaPts = coords.length > 1
    ? `${coords[0].x.toFixed(1)},${padT + plotH} ${polyPts} ${coords[coords.length - 1].x.toFixed(1)},${padT + plotH}`
    : "";

  const nodes = coords.map((c) => {
    const level = surpriseLevel(c.token.probability);
    const fill = level === "low" ? "var(--accent)" : level === "mid" ? "var(--warn)" : "var(--danger)";
    return `
      <circle class="spark-node" data-idx="${c.idx}" cx="${c.x.toFixed(1)}" cy="${c.y.toFixed(1)}" r="4.5" fill="${fill}" stroke="var(--surface)" stroke-width="1.5">
        <title>${escapeHtml(c.token.displaySymbol)}: ${c.bits.toFixed(2)} bits</title>
      </circle>
      <text x="${c.x.toFixed(1)}" y="${H - 5}" text-anchor="middle" class="svg-label">#${c.token.index}</text>
    `;
  }).join("");

  const activeCoord = coords[selectedTokenIdx] || coords[0];
  svg.innerHTML = `
    ${gridLines}
    ${areaPts ? `<polygon points="${areaPts}" fill="var(--accent-soft)" opacity="0.55"/>` : ""}
    ${coords.length > 1 ? `<polyline points="${polyPts}" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linejoin="round"/>` : ""}
    <circle id="spark-active-ring" cx="${activeCoord.x.toFixed(1)}" cy="${activeCoord.y.toFixed(1)}" r="8" fill="none" stroke="var(--cobalt)" stroke-width="2"/>
    ${nodes}
  `;

  svg.querySelectorAll(".spark-node").forEach((node) => {
    node.addEventListener("click", () => selectToken(Number(node.dataset.idx)));
  });
}

function selectToken(idx) {
  if (!currentAnalysis || !currentAnalysis.tokens[idx]) return;
  selectedTokenIdx = idx;
  const t = currentAnalysis.tokens[idx];

  document.querySelectorAll(".hm-token").forEach((el) => {
    el.classList.toggle("selected", Number(el.dataset.idx) === idx);
  });
  document.querySelectorAll("#tokens-table-body tr").forEach((el) => {
    el.classList.toggle("selected", Number(el.dataset.idx) === idx);
  });

  const activeNode = document.querySelector(`.spark-node[data-idx="${idx}"]`);
  const ring = document.getElementById("spark-active-ring");
  if (activeNode && ring) {
    ring.setAttribute("cx", activeNode.getAttribute("cx"));
    ring.setAttribute("cy", activeNode.getAttribute("cy"));
  }

  const bits = -Math.log2(Math.max(1e-12, t.probability));
  document.getElementById("insp-token").textContent = t.displaySymbol;
  document.getElementById("insp-id").textContent = String(t.tokenId);
  document.getElementById("insp-prob").textContent = `${(t.probability * 100).toFixed(4)}%`;
  document.getElementById("insp-inv").textContent = t.inverseProbability.toFixed(1);
  document.getElementById("insp-bits").textContent = bits.toFixed(2);
  document.getElementById("insp-ctx").textContent = t.context;

  let note = "";
  if (t.probability > 0.15) {
    note = `Highly predictable continuation after "${t.context}". Low surprise (${bits.toFixed(1)} bits) pulls perplexity down.`;
  } else if (t.probability > 0.01) {
    note = `Plausible continuation with moderate probability (${(t.probability * 100).toFixed(2)}%).`;
  } else {
    note = `Surprising token in this context! Its inverse penalty (1/P ≈ ${t.inverseProbability.toFixed(0)}) significantly raises overall perplexity.`;
  }
  document.getElementById("insp-explanation").textContent = note;
}

function renderGaugeAndMath(data) {
  const ppl = Number.isFinite(data.perplexity) ? data.perplexity : 10000;
  const ceNats = Number.isFinite(data.crossEntropy) ? data.crossEntropy : Math.log(ppl);
  const ceBits = ceNats / Math.LN2;

  document.getElementById("ppl-display").textContent = ppl >= 1000 ? ppl.toFixed(0) : ppl.toFixed(2);
  document.getElementById("ce-nats").textContent = ceNats.toFixed(3);
  document.getElementById("ce-bits").textContent = ceBits.toFixed(3);

  const norm = Math.max(0, Math.min(1, Math.log10(Math.max(1, ppl)) / 4));
  const arcLen = 301.6;
  document.getElementById("gauge-arc").setAttribute("stroke-dashoffset", String(arcLen * (1 - norm)));
  const angle = -90 + norm * 180;
  document.getElementById("gauge-needle").setAttribute("transform", `rotate(${angle.toFixed(1)}, 120, 116)`);

  const badge = document.getElementById("ppl-verdict-badge");
  if (ppl < 80) {
    badge.textContent = "Very Predictable";
    badge.style.color = "var(--accent)";
  } else if (ppl < 600) {
    badge.textContent = "Natural Phrasing";
    badge.style.color = "var(--cobalt)";
  } else if (ppl < 2500) {
    badge.textContent = "Unusual Context";
    badge.style.color = "var(--warn)";
  } else {
    badge.textContent = "High Surprise";
    badge.style.color = "var(--danger)";
  }

  const invs = data.tokens.map((t) => t.inverseProbability);
  const logProd = invs.reduce((acc, v) => acc + Math.log(v), 0);
  const approxProd = Math.exp(logProd);
  const prodLabel = approxProd > 1e8 ? approxProd.toExponential(2) : approxProd.toFixed(1);

  document.getElementById("math-step-1").textContent = `${invs.slice(0, 3).map((v) => v.toFixed(1)).join(" × ")}${invs.length > 3 ? " × …" : ""} = ${prodLabel}`;
  document.getElementById("math-step-2").textContent = `N = ${data.tokens.length}`;
  document.getElementById("math-step-3").textContent = `(${prodLabel})^(1/${data.tokens.length}) = ${ppl.toFixed(2)}`;
  document.getElementById("log-formula-note").textContent = `Log-space: exp(-1/${data.tokens.length} ∑ ln P) = exp(${ceNats.toFixed(3)}) = ${ppl.toFixed(2)}`;
}

function renderNextPredictions() {
  if (!currentAnalysis) return;
  const container = document.getElementById("top-next-container");
  const scopeNote = document.getElementById("temp-scope-note");
  const massBadge = document.getElementById("sampling-mass-badge");
  let items;

  const engine = loadedEngines[currentAnalysis.modelId];
  if (currentAnalysis.isLive && currentAnalysis.rawLastLogits && engine) {
    const fetchCount = Math.max(10, currentTopK);
    items = extractTopKFromLogits(currentAnalysis.rawLastLogits, engine.tokenizer, currentTemperature, fetchCount);
    scopeNote.textContent = `Live sampling across all 50,257 ${SUPPORTED_MODELS[currentAnalysis.modelId].short} logits (T=${currentTemperature.toFixed(2)}, K=${currentTopK}, P=${currentTopP.toFixed(2)}).`;
  } else {
    // Preserve exact verified probabilities at T = 1.00; scale proportionally when T != 1.00
    const rawSum = currentAnalysis.top8Next.reduce((acc, item) => acc + item.probability, 0);
    if (Math.abs(currentTemperature - 1.0) < 1e-3) {
      items = currentAnalysis.top8Next.map((item, idx) => ({ ...item, rank: idx + 1 }));
    } else {
      const logProbs = currentAnalysis.top8Next.map((item) => Math.log(Math.max(1e-9, item.probability)));
      const rescaled = stableSoftmax(logProbs, currentTemperature);
      items = currentAnalysis.top8Next
        .map((item, idx) => ({ ...item, probability: rescaled[idx] * rawSum }))
        .sort((a, b) => b.probability - a.probability)
        .map((item, idx) => ({ ...item, rank: idx + 1 }));
    }
    scopeNote.textContent = "Preview mode: sampling over top-8 snapshot. Run Analysis on custom text for full 50,257-vocab logits.";
  }

  let cumulative = 0;
  let keptMass = 0;
  let keptCount = 0;

  const annotated = items.map((item, idx) => {
    const withinK = idx < currentTopK;
    const withinP = idx === 0 || cumulative < currentTopP;
    cumulative += item.probability;
    const active = withinK && withinP;
    if (active) {
      keptMass += item.probability;
      keptCount++;
    }
    return { ...item, active };
  });

  massBadge.textContent = `Kept: ${(keptMass * 100).toFixed(1)}% · ${keptCount} tok`;

  container.innerHTML = "";
  const maxP = Math.max(...annotated.map((i) => i.probability), 0.01);

  annotated.slice(0, 12).forEach((item) => {
    const pct = (item.probability * 100).toFixed(2);
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `next-item ${item.rank === 1 && item.active ? "top-pick" : ""} ${item.active ? "" : "filtered-out"}`;
    btn.title = item.active
      ? `Append "${item.text}" to sentence and re-analyze`
      : "Excluded by current Top-K / Top-P cutoff";
    btn.innerHTML = `
      <span class="next-rank">#${item.rank}</span>
      <span class="next-mid">
        <span class="next-tok">${escapeHtml(item.displaySymbol)}</span>
        <span class="next-bar-track">
          <span class="next-bar-fill" style="width: ${(item.probability / maxP) * 100}%;"></span>
        </span>
      </span>
      <span class="next-pct">${pct}%</span>
    `;
    btn.addEventListener("click", () => {
      if (!item.active) return;
      const base = item.text.startsWith(" ") ? sentenceInput.value.replace(/[ \t]+$/, "") : sentenceInput.value;
      sentenceInput.value = base + item.text;
      document.querySelectorAll("#view-analyzer .pill-btn[data-sentence]").forEach((b) => b.classList.remove("active"));
      analyzeSentence(sentenceInput.value);
    });
    container.appendChild(btn);
  });
}

/* ============================================================================
   6. ANALYZER CONTROLS, AUTO-ANALYZE, COMPARE EXPERIMENTS, SHARE & EXPORT
   ============================================================================ */
tempSlider.addEventListener("input", () => {
  currentTemperature = parseFloat(tempSlider.value);
  document.getElementById("temp-val").textContent = `T = ${currentTemperature.toFixed(2)}`;
  renderNextPredictions();
});

topkSlider.addEventListener("input", () => {
  currentTopK = parseInt(topkSlider.value, 10);
  document.getElementById("topk-val").textContent = `K = ${currentTopK}`;
  renderNextPredictions();
});

toppSlider.addEventListener("input", () => {
  currentTopP = parseFloat(toppSlider.value);
  document.getElementById("topp-val").textContent = `P = ${currentTopP.toFixed(2)}`;
  renderNextPredictions();
});

document.getElementById("btn-calculate").addEventListener("click", () => {
  document.querySelectorAll("#view-analyzer .pill-btn[data-sentence]").forEach((b) => b.classList.remove("active"));
  analyzeSentence(sentenceInput.value);
});

sentenceInput.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    analyzeSentence(sentenceInput.value);
  }
});

let autoTimer = null;
sentenceInput.addEventListener("input", () => {
  if (!autoAnalyzeToggle.checked) return;
  const clean = sentenceInput.value.trim();
  if (!clean) return;
  if (!isActiveModelReady() && !(activeModelId === "Xenova/distilgpt2" && VERIFIED_SAMPLES[clean])) return;

  clearTimeout(autoTimer);
  autoTimer = setTimeout(() => analyzeSentence(clean), 450);
});

document.querySelectorAll("#view-analyzer .pill-btn[data-sentence]").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("#view-analyzer .pill-btn[data-sentence]").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    sentenceInput.value = btn.dataset.sentence;
    analyzeSentence(btn.dataset.sentence);
  });
});

async function runCompare(textA, textB, insight = "") {
  if (!textA || !textB) {
    showError("Enter both Sequence A and Sequence B to compare.");
    return;
  }
  clearError();
  setEngineStatus("analyzing", "Comparing A vs B...");
  try {
    const [resA, resB] = await Promise.all([evaluateSequence(textA), evaluateSequence(textB)]);
    renderAnalysis(resA);
    const ratio = resB.perplexity / resA.perplexity;
    const verdict = ratio >= 1
      ? `B is <strong>${ratio.toFixed(2)}×</strong> more surprising than A`
      : `A is <strong>${(1 / ratio).toFixed(2)}×</strong> more surprising than B`;

    compareResult.innerHTML = `
      <div><span class="muted-xs mono">SEQUENCE A PPL</span><br><strong class="mono">${resA.perplexity.toFixed(2)}</strong></div>
      <div><span class="muted-xs mono">SEQUENCE B PPL</span><br><strong class="mono">${resB.perplexity.toFixed(2)}</strong></div>
      <div><span class="muted-xs mono">COMPARISON</span><br><span>${verdict}${insight ? `<br><span class="muted-xs">${escapeHtml(insight)}</span>` : ""}</span></div>
    `;
    compareResult.hidden = false;
    updateModelHeaderUI();
  } catch (err) {
    showError(err.message);
  }
}

CONTEXT_EXPERIMENTS.forEach((exp) => {
  const pill = document.createElement("button");
  pill.type = "button";
  pill.className = "pill-btn";
  pill.textContent = exp.label;
  pill.title = exp.insight;
  pill.addEventListener("click", () => {
    sentenceInput.value = exp.a;
    sentenceInputB.value = exp.b;
    runCompare(exp.a, exp.b, exp.insight);
  });
  expRow.appendChild(pill);
});

btnToggleCompare.addEventListener("click", () => {
  const isOpen = !compareBox.hidden;
  compareBox.hidden = isOpen;
  btnToggleCompare.setAttribute("aria-pressed", String(!isOpen));
});

document.getElementById("btn-compare").addEventListener("click", () => {
  runCompare(sentenceInput.value.trim(), sentenceInputB.value.trim());
});

btnShareLink.addEventListener("click", async () => {
  const params = new URLSearchParams();
  params.set("a", sentenceInput.value);
  if (!compareBox.hidden && sentenceInputB.value.trim()) {
    params.set("b", sentenceInputB.value);
  }
  params.set("m", activeModelId);
  params.set("t", currentTemperature.toFixed(2));
  params.set("k", String(currentTopK));
  params.set("p", currentTopP.toFixed(2));

  const shareUrl = `${window.location.origin}${window.location.pathname}?${params.toString()}`;
  try {
    window.history.replaceState({}, "", `?${params.toString()}`);
  } catch {
    // Ignore history restrictions on opaque origins
  }

  try {
    if (navigator.clipboard && navigator.clipboard.writeText) {
      await navigator.clipboard.writeText(shareUrl);
      btnShareLink.textContent = "Copied Link ✓";
    } else {
      btnShareLink.textContent = "URL Updated ✓";
    }
  } catch {
    btnShareLink.textContent = "URL Updated ✓";
  }
  setTimeout(() => { btnShareLink.textContent = "🔗 Copy Share Link"; }, 2000);
});

function hydrateFromUrlParams() {
  const params = new URLSearchParams(window.location.search);
  if (params.has("m") && SUPPORTED_MODELS[params.get("m")]) {
    activeModelId = params.get("m");
    modelSelect.value = activeModelId;
  }
  if (params.has("a")) {
    sentenceInput.value = params.get("a");
  }
  if (params.has("b")) {
    sentenceInputB.value = params.get("b");
    compareBox.hidden = false;
    btnToggleCompare.setAttribute("aria-pressed", "true");
  }
  if (params.has("t")) {
    const t = Math.min(2.0, Math.max(0.2, parseFloat(params.get("t")) || 1.0));
    currentTemperature = t;
    tempSlider.value = String(t);
    document.getElementById("temp-val").textContent = `T = ${t.toFixed(2)}`;
  }
  if (params.has("k")) {
    const k = Math.min(15, Math.max(1, parseInt(params.get("k"), 10) || 8));
    currentTopK = k;
    topkSlider.value = String(k);
    document.getElementById("topk-val").textContent = `K = ${k}`;
  }
  if (params.has("p")) {
    const p = Math.min(1.0, Math.max(0.1, parseFloat(params.get("p")) || 1.0));
    currentTopP = p;
    toppSlider.value = String(p);
    document.getElementById("topp-val").textContent = `P = ${p.toFixed(2)}`;
  }
}

function downloadFile(filename, content, mimeType) {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

document.getElementById("btn-export-json").addEventListener("click", () => {
  if (!currentAnalysis) return;
  const payload = {
    model: currentAnalysis.modelId || activeModelId,
    sentence: currentAnalysis.sentence,
    perplexity: currentAnalysis.perplexity,
    crossEntropyNats: currentAnalysis.crossEntropy,
    tokens: currentAnalysis.tokens.map((t) => ({
      index: t.index,
      token: t.text,
      tokenId: t.tokenId,
      probability: t.probability,
      inverseProbability: t.inverseProbability,
      surprisalBits: -Math.log2(Math.max(1e-12, t.probability)),
      context: t.context
    }))
  };
  downloadFile("perplexity-analysis.json", JSON.stringify(payload, null, 2), "application/json");
});

document.getElementById("btn-export-csv").addEventListener("click", () => {
  if (!currentAnalysis) return;
  const header = "index,token,token_id,probability,inverse_prob,surprisal_bits\n";
  const rows = currentAnalysis.tokens.map((t) => {
    const bits = (-Math.log2(Math.max(1e-12, t.probability))).toFixed(4);
    const safeTok = `"${t.text.replace(/"/g, '""')}"`;
    return `${t.index},${safeTok},${t.tokenId},${t.probability.toExponential(6)},${t.inverseProbability.toFixed(2)},${bits}`;
  }).join("\n");
  downloadFile("perplexity-tokens.csv", header + rows, "text/csv");
});

/* ============================================================================
   7. VIEW 02: SANDBOX & BRANCHING VISUALIZER
   ============================================================================ */
let simTokens = SANDBOX_PRESETS["preset-a"].tokens.map((t) => ({ ...t }));

function renderSimTable() {
  const container = document.getElementById("sim-rows-container");
  container.innerHTML = "";

  let productInv = 1;
  let sumNegLog = 0;

  simTokens.forEach((t, idx) => {
    const p = Math.max(0.001, Math.min(1, t.prob));
    const inv = 1 / p;
    productInv *= inv;
    sumNegLog += -Math.log(p);

    const row = document.createElement("div");
    row.className = "sim-row";
    row.innerHTML = `
      <span class="mono muted-xs">#${idx + 1}</span>
      <input type="text" class="sim-input sim-name" data-idx="${idx}" value="${escapeHtml(t.text)}" aria-label="Token label">
      <input type="range" class="range-input sim-range" data-idx="${idx}" min="0.01" max="1.0" step="0.01" value="${p}">
      <input type="number" class="sim-input sim-num" data-idx="${idx}" min="0.001" max="1.0" step="0.01" value="${p}">
      <span class="mono muted-xs">1/P=${inv.toFixed(1)}</span>
      <button type="button" class="btn-del" data-idx="${idx}" title="Remove token">×</button>
    `;
    container.appendChild(row);
  });

  const n = simTokens.length;
  const arithPPL = Math.pow(productInv, 1 / n);
  const logPPL = Math.exp(sumNegLog / n);

  document.getElementById("sim-product-disp").textContent = productInv > 1e8 ? productInv.toExponential(2) : productInv.toFixed(2);
  document.getElementById("sim-power-disp").textContent = `1/${n} (N = ${n})`;
  document.getElementById("sim-ppl-result").textContent = arithPPL.toFixed(2);
  document.getElementById("sim-log-ppl-result").textContent = logPPL.toFixed(2);

  container.querySelectorAll(".sim-name").forEach((inp) => {
    inp.addEventListener("input", () => { simTokens[Number(inp.dataset.idx)].text = inp.value; });
  });
  container.querySelectorAll(".sim-range").forEach((rng) => {
    rng.addEventListener("input", () => {
      simTokens[Number(rng.dataset.idx)].prob = parseFloat(rng.value);
      renderSimTable();
    });
  });
  container.querySelectorAll(".sim-num").forEach((num) => {
    num.addEventListener("change", () => {
      let val = parseFloat(num.value);
      if (Number.isNaN(val) || val <= 0) val = 0.01;
      if (val > 1) val = 1;
      simTokens[Number(num.dataset.idx)].prob = val;
      renderSimTable();
    });
  });
  container.querySelectorAll(".btn-del").forEach((btn) => {
    btn.addEventListener("click", () => {
      if (simTokens.length <= 1) return;
      simTokens.splice(Number(btn.dataset.idx), 1);
      renderSimTable();
    });
  });
}

document.getElementById("btn-add-sim-token").addEventListener("click", () => {
  simTokens.push({ text: `tok_${simTokens.length + 1}`, prob: 0.5 });
  renderSimTable();
});

Object.keys(SANDBOX_PRESETS).forEach((presetId) => {
  document.getElementById(presetId).addEventListener("click", () => {
    document.querySelectorAll(".preset-chip").forEach((c) => c.classList.remove("active"));
    document.getElementById(presetId).classList.add("active");
    simTokens = SANDBOX_PRESETS[presetId].tokens.map((t) => ({ ...t }));
    document.getElementById("sim-callout-explanation").innerHTML = SANDBOX_PRESETS[presetId].note;
    renderSimTable();
  });
});

function buildCurvePath() {
  const pts = [];
  for (let p = 0.05; p <= 1.001; p += 0.025) {
    const x = 40 + ((p - 0.05) / 0.95) * 300;
    const ppl = Math.min(20, 1 / p);
    const y = 155 - ((ppl - 1) / 19) * 135;
    pts.push(`${pts.length === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`);
  }
  curvePath.setAttribute("d", pts.join(" "));
}

function updateBranchingVisualizer() {
  const p = parseFloat(visualSlider.value);
  const ppl = 1 / p;
  const bits = -Math.log2(p);

  document.getElementById("visual-slider-p").textContent = p.toFixed(2);
  document.getElementById("visual-slider-ppl").textContent = ppl.toFixed(2);
  document.getElementById("visual-slider-bits").textContent = bits.toFixed(2);
  document.getElementById("visual-slider-choices").textContent = ppl.toFixed(1);

  const count = Math.min(50, Math.max(1, Math.round(ppl)));
  const dotsWrap = document.getElementById("branch-dots");
  dotsWrap.innerHTML = "";
  for (let i = 0; i < count; i++) {
    const dot = document.createElement("span");
    dot.className = `branch-dot ${i === 0 ? "picked" : ""}`;
    dotsWrap.appendChild(dot);
  }

  const clampedP = Math.max(0.05, p);
  const x = 40 + ((clampedP - 0.05) / 0.95) * 300;
  const y = 155 - ((Math.min(20, ppl) - 1) / 19) * 135;
  curveDot.setAttribute("cx", x.toFixed(1));
  curveDot.setAttribute("cy", y.toFixed(1));
}

visualSlider.addEventListener("input", updateBranchingVisualizer);

/* ============================================================================
   8. VIEW 03: BPE TOKENIZER X-RAY & CONCEPTS
   ============================================================================ */
function heuristicTokenize(text) {
  const pattern = /'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+/g;
  const matches = text.match(pattern) || [];
  return matches.map((tok) => ({ text: tok, id: null }));
}

function updateXRay() {
  const val = xrayInput.value;
  if (!val) {
    xrayBadge.textContent = getActiveEngine() ? `Exact ${SUPPORTED_MODELS[activeModelId].short} BPE` : "Heuristic BPE Preview";
    xrayTokens.innerHTML = "";
    xrayStats.textContent = "0 tokens · 0 characters · 0.00 chars/token";
    return;
  }

  let pieces = [];
  const engine = getActiveEngine();

  if (engine) {
    xrayBadge.textContent = `Exact ${SUPPORTED_MODELS[activeModelId].short} BPE`;
    const encoded = engine.tokenizer(val);
    const ids = Array.from(encoded.input_ids.data).map(Number);
    pieces = ids.map((id) => ({ text: engine.tokenizer.decode([id]), id }));
  } else {
    xrayBadge.textContent = "Heuristic BPE Preview";
    pieces = heuristicTokenize(val);
  }

  xrayTokens.innerHTML = "";
  pieces.forEach((p) => {
    const chip = document.createElement("span");
    chip.className = "xray-chip";
    chip.innerHTML = `
      <span>${escapeHtml(formatDisplayToken(p.text))}</span>
      ${p.id !== null ? `<span class="xray-id">${p.id}</span>` : ""}
    `;
    xrayTokens.appendChild(chip);
  });

  const chars = val.length;
  const ratio = pieces.length > 0 ? (chars / pieces.length).toFixed(2) : "0.00";
  xrayStats.textContent = `${pieces.length} tokens · ${chars} characters · ${ratio} chars/token`;
}

xrayInput.addEventListener("input", updateXRay);

function renderConcepts() {
  const grid = document.getElementById("concepts-grid");
  grid.innerHTML = "";
  CONCEPTS_DATA.forEach((c) => {
    const card = document.createElement("article");
    card.className = "concept-card";
    card.innerHTML = `
      <span class="concept-tag">${c.tag}</span>
      <h3>${c.title}</h3>
      <div class="concept-formula">${c.formula}</div>
      <p>${c.body}</p>
    `;
    grid.appendChild(card);
  });
}

/* ============================================================================
   9. VIEW 04: KNOWLEDGE CHECK (QUIZ)
   ============================================================================ */
let qIdx = 0;
let qScore = 0;
let qStreak = 0;
let qAnswered = false;

function renderQuizQuestion() {
  qAnswered = false;
  const q = QUIZ_DATA[qIdx];
  qTextEl.textContent = q.q;
  qProgressText.textContent = `Question ${qIdx + 1} of ${QUIZ_DATA.length}`;
  qProgressFill.style.width = `${((qIdx + 1) / QUIZ_DATA.length) * 100}%`;
  qFeedbackEl.hidden = true;
  btnNextQ.hidden = true;

  qOptsEl.innerHTML = "";
  q.options.forEach((opt, idx) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "quiz-opt";
    btn.innerHTML = `
      <span class="opt-letter">${String.fromCharCode(65 + idx)}</span>
      <span>${escapeHtml(opt)}</span>
    `;
    btn.addEventListener("click", () => handleQuizAnswer(idx, btn));
    qOptsEl.appendChild(btn);
  });
}

function handleQuizAnswer(selectedIdx, btn) {
  if (qAnswered) return;
  qAnswered = true;
  const q = QUIZ_DATA[qIdx];
  const allOpts = qOptsEl.querySelectorAll(".quiz-opt");
  allOpts.forEach((o) => { o.disabled = true; });

  if (selectedIdx === q.correct) {
    btn.classList.add("correct");
    qScore++;
    qStreak++;
    qFeedbackEl.innerHTML = `<strong style="color: var(--accent);">Correct.</strong> ${q.explanation}`;
  } else {
    btn.classList.add("incorrect");
    allOpts[q.correct].classList.add("correct");
    qStreak = 0;
    qFeedbackEl.innerHTML = `<strong style="color: var(--danger);">Not quite.</strong> ${q.explanation}`;
  }

  qScoreEl.textContent = String(qScore);
  qStreakEl.textContent = String(qStreak);
  qFeedbackEl.hidden = false;
  btnNextQ.hidden = false;
  btnNextQ.textContent = qIdx === QUIZ_DATA.length - 1 ? "See Final Score →" : "Next Question →";
}

btnNextQ.addEventListener("click", () => {
  if (qIdx < QUIZ_DATA.length - 1) {
    qIdx++;
    renderQuizQuestion();
  } else {
    qTextEl.textContent = `Completed! You scored ${qScore} / ${QUIZ_DATA.length}.`;
    qOptsEl.innerHTML = `
      <div class="callout-info">
        ${qScore >= 7
          ? "Excellent mastery of autoregressive probabilities, cross-entropy, and perplexity."
          : "Good run! Experiment with the Live Analyzer and Sandbox to sharpen any missed concepts."}
      </div>
    `;
    qFeedbackEl.hidden = true;
    btnNextQ.hidden = true;
  }
});

document.getElementById("btn-restart-quiz").addEventListener("click", () => {
  qIdx = 0;
  qScore = 0;
  qStreak = 0;
  qScoreEl.textContent = "0";
  qStreakEl.textContent = "0";
  renderQuizQuestion();
});

/* ============================================================================
   10. VIEW 05: HUMAN VS. AI AUTOCOMPLETE RACE
   ============================================================================ */
const MAX_RACE_ROUNDS = 5;
let raceContext = RACE_PROMPTS[0].prefix;
let raceRounds = [];

RACE_PROMPTS.forEach((rp, idx) => {
  const pill = document.createElement("button");
  pill.type = "button";
  pill.className = `pill-btn ${idx === 0 ? "active" : ""}`;
  pill.textContent = rp.label;
  pill.addEventListener("click", () => {
    racePresetsEl.querySelectorAll(".pill-btn").forEach((b) => b.classList.remove("active"));
    pill.classList.add("active");
    racePrefixInput.value = rp.prefix;
    resetRace(rp.prefix);
  });
  racePresetsEl.appendChild(pill);
});

function resetRace(prefixText = racePrefixInput.value.trim()) {
  raceContext = prefixText || RACE_PROMPTS[0].prefix;
  raceRounds = [];
  raceContextDisplay.textContent = raceContext;
  raceGuessInput.value = "";
  raceGuessInput.disabled = false;
  document.getElementById("btn-race-submit").disabled = false;
  raceRoundBadge.textContent = `Round 1 / ${MAX_RACE_ROUNDS}`;
  raceHumanPplEl.textContent = "—";
  raceAiPplEl.textContent = "—";
  raceFeedbackEl.hidden = true;
  raceHistoryEl.innerHTML = `<div class="muted-xs">Submit your first guess to score Round 1 against ${SUPPORTED_MODELS[activeModelId].short}.</div>`;
}

document.getElementById("btn-race-start").addEventListener("click", () => {
  resetRace(racePrefixInput.value.trim());
});

document.getElementById("btn-race-reset").addEventListener("click", () => {
  resetRace(racePrefixInput.value.trim());
});

document.getElementById("race-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (raceRounds.length >= MAX_RACE_ROUNDS) return;

  const rawGuess = raceGuessInput.value;
  if (!rawGuess.trim()) return;

  clearError();
  const submitBtn = document.getElementById("btn-race-submit");
  submitBtn.disabled = true;
  setEngineStatus("analyzing", "Scoring guess...");

  try {
    const ready = await ensureModelLoaded(activeModelId);
    if (!ready) {
      submitBtn.disabled = false;
      return;
    }

    const engine = getActiveEngine();
    const { tokenizer } = engine;

    const needsSpace = !/\s$/.test(raceContext) && !/^[\s.,!?;:)'"\]]/.test(rawGuess);
    const formattedGuess = needsSpace ? ` ${rawGuess.trim()}` : rawGuess;

    const ctxEncoded = tokenizer(raceContext);
    const ctxIds = Array.from(ctxEncoded.input_ids.data).map(Number);
    const guessEncoded = tokenizer(formattedGuess);
    const guessIds = Array.from(guessEncoded.input_ids.data).map(Number);

    if (ctxIds.length === 0) throw new Error("Starter prefix produced zero tokens.");
    if (guessIds.length === 0) throw new Error("Guess produced zero tokens.");

    const { logitsData, vocabSize } = await forwardCausalLM(engine, ctxIds, activeModelId);
    const lastRow = logitsData.subarray(ctxIds.length * vocabSize, (ctxIds.length + 1) * vocabSize);
    const probs = stableSoftmax(lastRow, 1.0);

    const firstGuessId = guessIds[0];
    const firstGuessText = tokenizer.decode([firstGuessId]);
    const humanProb = Math.max(1e-12, probs[firstGuessId]);

    const top5 = extractTopKFromLogits(lastRow, tokenizer, 1.0, 5);
    const aiTop = top5[0];

    raceRounds.push({
      round: raceRounds.length + 1,
      humanToken: firstGuessText,
      humanProb,
      aiToken: aiTop.text,
      aiProb: aiTop.probability
    });

    raceContext += firstGuessText;
    raceContextDisplay.textContent = raceContext;
    raceGuessInput.value = "";

    const humanPPL = Math.exp(
      raceRounds.reduce((s, r) => s - Math.log(r.humanProb), 0) / raceRounds.length
    );
    const aiPPL = Math.exp(
      raceRounds.reduce((s, r) => s - Math.log(r.aiProb), 0) / raceRounds.length
    );

    raceHumanPplEl.textContent = humanPPL >= 1000 ? humanPPL.toFixed(0) : humanPPL.toFixed(2);
    raceAiPplEl.textContent = aiPPL.toFixed(2);

    const top5Html = top5.map((t) => `<code>${escapeHtml(formatDisplayToken(t.text))} (${(t.probability * 100).toFixed(1)}%)</code>`).join(" ");
    raceFeedbackEl.innerHTML = `
      <div><strong>Round ${raceRounds.length}:</strong> Your token <code>${escapeHtml(formatDisplayToken(firstGuessText))}</code> had probability <strong>${(humanProb * 100).toFixed(2)}%</strong> · AI #1 was <code>${escapeHtml(formatDisplayToken(aiTop.text))}</code> (<strong>${(aiTop.probability * 100).toFixed(2)}%</strong>).</div>
      <div style="margin-top: 8px;"><span class="muted-xs mono">AI TOP-5:</span> ${top5Html}</div>
    `;
    raceFeedbackEl.hidden = false;

    raceHistoryEl.innerHTML = raceRounds.map((r) => `
      <div class="race-round-item">
        <div class="mono muted-xs">Round ${r.round}</div>
        <div>You: <code>${escapeHtml(formatDisplayToken(r.humanToken))}</code> (${(r.humanProb * 100).toFixed(2)}%) · AI: <code>${escapeHtml(formatDisplayToken(r.aiToken))}</code> (${(r.aiProb * 100).toFixed(2)}%)</div>
      </div>
    `).join("");

    if (raceRounds.length >= MAX_RACE_ROUNDS) {
      raceRoundBadge.textContent = "Race Complete";
      raceGuessInput.disabled = true;
      submitBtn.disabled = true;
      raceFeedbackEl.innerHTML += `<div class="callout-info" style="margin-top: 10px;"><strong>Final Result:</strong> Human PPL = <strong>${humanPPL.toFixed(2)}</strong> vs. AI Top-1 PPL = <strong>${aiPPL.toFixed(2)}</strong>.</div>`;
    } else {
      raceRoundBadge.textContent = `Round ${raceRounds.length + 1} / ${MAX_RACE_ROUNDS}`;
      submitBtn.disabled = false;
      raceGuessInput.focus();
    }

    updateModelHeaderUI();
  } catch (err) {
    showError(`Race step failed: ${err.message}`);
    submitBtn.disabled = false;
  }
});

/* ============================================================================
   11. BOOTSTRAP INITIAL STATE
   ============================================================================ */
hydrateFromUrlParams();
updateModelHeaderUI();

const initialText = sentenceInput.value.trim();
if (activeModelId === "Xenova/distilgpt2" && VERIFIED_SAMPLES[initialText]) {
  renderAnalysis({ ...VERIFIED_SAMPLES[initialText], modelId: "Xenova/distilgpt2", sentence: initialText, isLive: false });
} else if (initialText) {
  renderAnalysis({ ...VERIFIED_SAMPLES["I love Python"], modelId: "Xenova/distilgpt2", sentence: "I love Python", isLive: false });
  analyzeSentence(initialText);
} else {
  renderAnalysis({ ...VERIFIED_SAMPLES["I love Python"], modelId: "Xenova/distilgpt2", sentence: "I love Python", isLive: false });
}

renderSimTable();
buildCurvePath();
updateBranchingVisualizer();
updateXRay();
renderConcepts();
renderQuizQuestion();
