import { VERIFIED_SAMPLES, SANDBOX_PRESETS, QUIZ_DATA, CONCEPTS_DATA } from "./data.js";

/* ============================================================================
   1. INITIALIZATION, THEME & NAVIGATION
   ============================================================================ */
const protocolBanner = document.getElementById("protocol-banner");
if (window.location.protocol === "file:") {
  protocolBanner.hidden = false;
}

const themeToggleBtn = document.getElementById("theme-toggle");
const themeIcon = document.getElementById("theme-icon");

function applyTheme(theme) {
  document.documentElement.setAttribute("data-theme", theme);
  themeIcon.textContent = theme === "dark" ? "☀" : "☾";
  localStorage.setItem("perplexity-lab-theme", theme);
}

applyTheme(localStorage.getItem("perplexity-lab-theme") || "light");

themeToggleBtn.addEventListener("click", () => {
  const next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
  applyTheme(next);
});

const viewTabs = document.querySelectorAll(".view-tab");
const viewPanels = document.querySelectorAll(".view-panel");

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
   2. MODEL ENGINE (LAZY DYNAMIC TRANSFORMERS.JS DISTILGPT-2)
   ============================================================================ */
let transformersMod = null;
let tokenizer = null;
let model = null;
let isModelLoading = false;
let isModelReady = false;

let currentAnalysis = null;
let selectedTokenIdx = 0;
let currentTemperature = 1.0;

const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const modelCallout = document.getElementById("model-callout");
const calloutText = document.getElementById("callout-text");
const calloutPct = document.getElementById("callout-pct");
const loadProgressFill = document.getElementById("load-progress-fill");
const errorBox = document.getElementById("error-box");

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

async function ensureModelLoaded() {
  if (isModelReady) return true;
  if (isModelLoading) return false;

  isModelLoading = true;
  setEngineStatus("loading", "Loading DistilGPT-2...");
  modelCallout.hidden = false;
  setProgress(8, "Loading Transformers.js runtime...");

  try {
    transformersMod = await import("https://cdn.jsdelivr.net/npm/@huggingface/transformers@3");
    const { AutoTokenizer, AutoModelForCausalLM } = transformersMod;

    setProgress(18, "Downloading DistilGPT-2 BPE tokenizer...");
    tokenizer = await AutoTokenizer.from_pretrained("Xenova/distilgpt2", {
      progress_callback: (p) => {
        if (p.status === "progress" && typeof p.progress === "number") {
          setProgress(18 + p.progress * 0.22);
        }
      }
    });

    setProgress(42, "Downloading DistilGPT-2 quantized ONNX weights...");
    model = await AutoModelForCausalLM.from_pretrained("Xenova/distilgpt2", {
      dtype: "q8",
      progress_callback: (p) => {
        if (p.status === "progress" && typeof p.progress === "number") {
          setProgress(42 + p.progress * 0.56);
        }
      }
    });

    setProgress(100, "DistilGPT-2 ready in browser.");
    isModelReady = true;
    isModelLoading = false;
    setEngineStatus("ready", "DistilGPT-2 Ready");
    updateXRay();
    setTimeout(() => {
      modelCallout.hidden = true;
    }, 2600);
    return true;
  } catch (err) {
    console.error("Model load error:", err);
    isModelLoading = false;
    setEngineStatus("error", "Model Load Failed");
    modelCallout.hidden = true;
    showError(`Could not load in-browser DistilGPT-2 (${err.message}). Verified sample snapshots remain available.`);
    return false;
  }
}

function showError(msg) {
  errorBox.textContent = msg;
  errorBox.hidden = false;
}

function clearError() {
  errorBox.hidden = true;
  errorBox.textContent = "";
}

/* ============================================================================
   3. MATH HELPERS & CAUSAL INFERENCE
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

async function runLiveInference(sentenceText) {
  const { Tensor } = transformersMod;
  const encoded = tokenizer(sentenceText);
  const rawIds = Array.from(encoded.input_ids.data).map(Number);

  if (rawIds.length === 0) {
    throw new Error("Input produced zero tokens.");
  }

  const bosId = 50256; // <|endoftext|>
  const inputIds = [bosId, ...rawIds];

  const tensorInput = new Tensor("int64", BigInt64Array.from(inputIds.map(BigInt)), [1, inputIds.length]);
  const tensorMask = new Tensor("int64", BigInt64Array.from(inputIds.map(() => 1n)), [1, inputIds.length]);

  const out = await model({ input_ids: tensorInput, attention_mask: tensorMask });
  const vocabSize = 50257;
  const logitsData = out.logits.data;

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
  const top8Next = extractTopKFromLogits(lastLogits, 1.0, 8);

  const crossEntropy = -sumLogProbs / rawIds.length;
  const perplexity = Math.exp(crossEntropy);

  return {
    sentence: sentenceText,
    tokens,
    perplexity,
    crossEntropy,
    top8Next,
    rawLastLogits: lastLogits,
    isLive: true
  };
}

function extractTopKFromLogits(logits, temperature, k = 8) {
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

async function evaluateSequence(text, allowPreviewFallback = true) {
  const clean = text.trim();
  if (!clean) throw new Error("Please enter a non-empty sentence.");

  if (!isModelReady && allowPreviewFallback && VERIFIED_SAMPLES[clean]) {
    return { ...VERIFIED_SAMPLES[clean], sentence: clean, isLive: false };
  }

  const loaded = await ensureModelLoaded();
  if (!loaded) {
    if (VERIFIED_SAMPLES[clean]) {
      return { ...VERIFIED_SAMPLES[clean], sentence: clean, isLive: false };
    }
    throw new Error("Live DistilGPT-2 model is unavailable.");
  }

  return await runLiveInference(clean);
}

async function analyzeSentence(sentenceText) {
  const clean = sentenceText.trim();
  if (!clean) {
    showError("Please enter a sentence to evaluate.");
    return;
  }
  clearError();

  // Instant verified snapshot preview if model is not loaded yet
  if (!isModelReady && VERIFIED_SAMPLES[clean]) {
    renderAnalysis({ ...VERIFIED_SAMPLES[clean], sentence: clean, isLive: false });
    return;
  }

  setEngineStatus("analyzing", "Running inference...");
  try {
    const result = await evaluateSequence(clean, false);
    renderAnalysis(result);
    setEngineStatus("ready", "DistilGPT-2 Ready");
  } catch (err) {
    if (!isModelReady) setEngineStatus("error", "Model Unavailable");
    showError(err.message);
  }
}

/* ============================================================================
   4. RENDER LIVE ANALYZER WORKBENCH
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

    // Heatmap chip
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

    // Table row
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
  const ppl = data.perplexity;
  const ceNats = data.crossEntropy;
  const ceBits = ceNats / Math.LN2;

  document.getElementById("ppl-display").textContent = ppl >= 1000 ? ppl.toFixed(0) : ppl.toFixed(2);
  document.getElementById("ce-nats").textContent = ceNats.toFixed(3);
  document.getElementById("ce-bits").textContent = ceBits.toFixed(3);

  // Logarithmic gauge mapping: PPL 1 -> 0, PPL 10000 -> 1
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

  // 3-step breakdown
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
  let items;

  if (currentAnalysis.isLive && currentAnalysis.rawLastLogits) {
    items = extractTopKFromLogits(currentAnalysis.rawLastLogits, currentTemperature, 8);
    scopeNote.textContent = `Live temperature scaling across all 50,257 DistilGPT-2 vocabulary logits (T = ${currentTemperature.toFixed(2)}).`;
  } else {
    // Scale preview top-8 probabilities via log-probs / T
    const logProbs = currentAnalysis.top8Next.map((item) => Math.log(Math.max(1e-9, item.probability)));
    const rescaled = stableSoftmax(logProbs, currentTemperature);
    items = currentAnalysis.top8Next.map((item, idx) => ({
      ...item,
      probability: rescaled[idx]
    })).sort((a, b) => b.probability - a.probability)
      .map((item, idx) => ({ ...item, rank: idx + 1 }));
    scopeNote.textContent = "Preview mode: temperature rescales top-8 snapshot. Click Run Analysis on custom text for full 50,257-vocab logits.";
  }

  container.innerHTML = "";
  const maxP = Math.max(...items.map((i) => i.probability), 0.01);

  items.forEach((item) => {
    const pct = (item.probability * 100).toFixed(2);
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = `next-item ${item.rank === 1 ? "top-pick" : ""}`;
    btn.title = `Append "${item.text}" to sentence and re-analyze`;
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
      const input = document.getElementById("sentence-input");
      input.value = input.value + item.text;
      document.querySelectorAll(".pill-btn").forEach((b) => b.classList.remove("active"));
      analyzeSentence(input.value);
    });
    container.appendChild(btn);
  });
}

// Analyzer Controls
const sentenceInput = document.getElementById("sentence-input");
const tempSlider = document.getElementById("temp-slider");
const tempVal = document.getElementById("temp-val");

tempSlider.addEventListener("input", () => {
  currentTemperature = parseFloat(tempSlider.value);
  tempVal.textContent = `T = ${currentTemperature.toFixed(2)}`;
  renderNextPredictions();
});

document.getElementById("btn-calculate").addEventListener("click", () => {
  document.querySelectorAll(".pill-btn").forEach((b) => b.classList.remove("active"));
  analyzeSentence(sentenceInput.value);
});

sentenceInput.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key === "Enter") {
    analyzeSentence(sentenceInput.value);
  }
});

document.querySelectorAll(".pill-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".pill-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    sentenceInput.value = btn.dataset.sentence;
    analyzeSentence(btn.dataset.sentence);
  });
});

// Compare A vs B
const btnToggleCompare = document.getElementById("btn-toggle-compare");
const compareBox = document.getElementById("compare-box");
const compareResult = document.getElementById("compare-result");

btnToggleCompare.addEventListener("click", () => {
  const isOpen = !compareBox.hidden;
  compareBox.hidden = isOpen;
  btnToggleCompare.setAttribute("aria-pressed", String(!isOpen));
});

document.getElementById("btn-compare").addEventListener("click", async () => {
  const textA = sentenceInput.value.trim();
  const textB = document.getElementById("sentence-input-b").value.trim();
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
      <div><span class="muted-xs mono">COMPARISON</span><br><span>${verdict}</span></div>
    `;
    compareResult.hidden = false;
    setEngineStatus(isModelReady ? "ready" : "", isModelReady ? "DistilGPT-2 Ready" : "Preview Mode");
  } catch (err) {
    showError(err.message);
  }
});

/* ============================================================================
   5. VIEW 02: SANDBOX & BRANCHING VISUALIZER
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
    inp.addEventListener("input", (e) => {
      simTokens[Number(e.target.dataset.idx)].text = e.target.value;
    });
  });

  container.querySelectorAll(".sim-range").forEach((rng) => {
    rng.addEventListener("input", (e) => {
      simTokens[Number(e.target.dataset.idx)].prob = parseFloat(e.target.value);
      renderSimTable();
    });
  });

  container.querySelectorAll(".sim-num").forEach((num) => {
    num.addEventListener("change", (e) => {
      let val = parseFloat(e.target.value);
      if (Number.isNaN(val) || val <= 0) val = 0.01;
      if (val > 1) val = 1;
      simTokens[Number(e.target.dataset.idx)].prob = val;
      renderSimTable();
    });
  });

  container.querySelectorAll(".btn-del").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      if (simTokens.length <= 1) return;
      simTokens.splice(Number(e.target.dataset.idx), 1);
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

// Branching Factor SVG Curve & Dots
const visualSlider = document.getElementById("visual-slider");
const curvePath = document.getElementById("ppl-curve-path");
const curveDot = document.getElementById("ppl-curve-dot");

function buildCurvePath() {
  // X: P from 0.05..1.00 mapped to 40..340; Y: PPL = 1/P capped at 20 mapped to 155..20
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
   6. VIEW 03: BPE TOKENIZER X-RAY & CONCEPTS
   ============================================================================ */
const xrayInput = document.getElementById("xray-input");
const xrayTokens = document.getElementById("xray-tokens");
const xrayStats = document.getElementById("xray-stats");
const xrayBadge = document.getElementById("xray-mode-badge");

function heuristicTokenize(text) {
  // Approximates GPT-2 pre-tokenization boundaries when model is not yet loaded
  const pattern = /'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?[0-9]+| ?[^\sA-Za-z0-9]+|\s+/g;
  const matches = text.match(pattern) || [];
  return matches.map((tok) => ({
    text: tok,
    id: null
  }));
}

function updateXRay() {
  const val = xrayInput.value;
  let pieces = [];

  if (isModelReady && tokenizer) {
    xrayBadge.textContent = "Exact DistilGPT-2 BPE";
    const encoded = tokenizer(val);
    const ids = Array.from(encoded.input_ids.data).map(Number);
    pieces = ids.map((id) => ({
      text: tokenizer.decode([id]),
      id
    }));
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
   7. VIEW 04: KNOWLEDGE CHECK (QUIZ)
   ============================================================================ */
let qIdx = 0;
let qScore = 0;
let qStreak = 0;
let qAnswered = false;

const qTextEl = document.getElementById("quiz-question");
const qOptsEl = document.getElementById("quiz-options");
const qFeedbackEl = document.getElementById("quiz-feedback");
const qProgressText = document.getElementById("quiz-progress-text");
const qProgressFill = document.getElementById("quiz-progress-fill");
const qScoreEl = document.getElementById("quiz-score");
const qStreakEl = document.getElementById("quiz-streak");
const btnNextQ = document.getElementById("btn-next-question");
const btnRestartQ = document.getElementById("btn-restart-quiz");

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

btnRestartQ.addEventListener("click", () => {
  qIdx = 0;
  qScore = 0;
  qStreak = 0;
  qScoreEl.textContent = "0";
  qStreakEl.textContent = "0";
  renderQuizQuestion();
});

/* ============================================================================
   8. BOOTSTRAP INITIAL STATE
   ============================================================================ */
renderAnalysis({ ...VERIFIED_SAMPLES["I love Python"], sentence: "I love Python", isLive: false });
renderSimTable();
buildCurvePath();
updateBranchingVisualizer();
updateXRay();
renderConcepts();
renderQuizQuestion();
