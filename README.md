# Perplexity Lab — Interactive Token Probability & Entropy Observatory

An interactive web workbench for exploring how autoregressive language models (DistilGPT-2 running locally in the browser) assign subword token probabilities, compute cross-entropy in nats and bits, and measure perplexity.

## Architecture & Files

- `index.html` — 4-view split-workspace interface (`Live Analyzer`, `Math & Sandbox`, `BPE & Concepts`, `Knowledge Check`)
- `styles.css` — Editorial Observatory theme with warm paper light mode and obsidian dark mode toggle
- `app.js` — Lazy-loaded `@huggingface/transformers@3` (`Xenova/distilgpt2` Q8 ONNX) inference, temperature logit scaling, A/B sequence comparator, SVG perplexity gauge, branching visualizer, BPE X-ray, and quiz engine
- `data.js` — Verified DistilGPT-2 sample snapshots, sandbox presets, concept cards, and quiz questions

## Run Locally

Because the app uses ES modules (`<script type="module">`), serve it with any local static server:

```bash
python3 -m http.server 8080
```

Then open `http://localhost:8080` in your browser.
