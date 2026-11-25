# Perplexity Lab — Interactive Token Probability & Entropy Observatory

An interactive web workbench for exploring how autoregressive language models (`DistilGPT-2` and `GPT-2 Small` running locally in the browser) assign subword token probabilities, compute cross-entropy in nats and bits, and measure perplexity.

## Key Capabilities

- **01 · Live Analyzer**: Token surprise heatmap, clickable **Surprisal Trajectory SVG Sparkline** (`−log₂ P` bits/token), inline token inspector, semicircular perplexity gauge, **Temperature + Top-K + Top-P (Nucleus) decoding visualizer**, click-to-append next token, **Compare A vs B** with one-click context experiments, debounced **Auto-analyze**, shareable URL state, and JSON/CSV export.
- **02 · Math & Sandbox**: Manual probability table comparing arithmetic vs. log-space perplexity, 4 presets, and a live `1/P` branching-factor curve and dot visualizer.
- **03 · BPE & Concepts**: Live tokenizer X-ray (upgrading from heuristic preview to exact 50,257-vocab GPT-2 BPE once loaded) and concept reference cards.
- **04 · Knowledge Check**: 8-question interactive quiz with streak tracking and explanations.
- **05 · Human vs. AI Race**: 5-round next-token prediction challenge comparing your running Human Perplexity against the model's Top-1 Perplexity.

## Run Locally

Because the app uses ES modules (`<script type="module">`), serve it with any local static server:

```bash
python3 -m http.server 8080
```

Then open `http://localhost:8080` in your browser.
