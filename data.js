/**
 * Verified DistilGPT-2 sample snapshots, model metadata, context experiments,
 * sandbox presets, race prompts, quiz questions, and conceptual glossary.
 */

export const SUPPORTED_MODELS = {
  "Xenova/distilgpt2": {
    id: "Xenova/distilgpt2",
    short: "DistilGPT-2",
    label: "DistilGPT-2 (82M · 6 layers)",
    vocabSize: 50257
  },
  "Xenova/gpt2": {
    id: "Xenova/gpt2",
    short: "GPT-2 Small",
    label: "GPT-2 Small (124M · 12 layers)",
    vocabSize: 50257
  }
};

export const VERIFIED_SAMPLES = {
  "I love Python": {
    tokens: [
      { index: 1, text: "I", displaySymbol: "I", tokenId: 40, probability: 0.007748, inverseProbability: 129.07, context: "[BOS]" },
      { index: 2, text: " love", displaySymbol: "␣love", tokenId: 1842, probability: 0.016743, inverseProbability: 59.73, context: "[BOS] I" },
      { index: 3, text: " Python", displaySymbol: "␣Python", tokenId: 11361, probability: 0.000061, inverseProbability: 16402.49, context: "[BOS] I ␣love" }
    ],
    perplexity: 501.92,
    crossEntropy: 6.218,
    top8Next: [
      { rank: 1, tokenId: 13, text: ".", displaySymbol: ".", probability: 0.2462 },
      { rank: 2, tokenId: 11, text: ",", displaySymbol: ",", probability: 0.1629 },
      { rank: 3, tokenId: 290, text: " and", displaySymbol: "␣and", probability: 0.1339 },
      { rank: 4, tokenId: 0, text: "!", displaySymbol: "!", probability: 0.0814 },
      { rank: 5, tokenId: 513, text: " 3", displaySymbol: "␣3", probability: 0.0282 },
      { rank: 6, tokenId: 362, text: " 2", displaySymbol: "␣2", probability: 0.0192 },
      { rank: 7, tokenId: 284, text: " to", displaySymbol: "␣to", probability: 0.0172 },
      { rank: 8, tokenId: 329, text: " for", displaySymbol: "␣for", probability: 0.0139 }
    ]
  },
  "Python love I": {
    tokens: [
      { index: 1, text: "Python", displaySymbol: "Python", tokenId: 37906, probability: 0.000065, inverseProbability: 15431.39, context: "[BOS]" },
      { index: 2, text: " love", displaySymbol: "␣love", tokenId: 1842, probability: 0.000051, inverseProbability: 19634.79, context: "[BOS] Python" },
      { index: 3, text: " I", displaySymbol: "␣I", tokenId: 314, probability: 0.000574, inverseProbability: 1743.07, context: "[BOS] Python ␣love" }
    ],
    perplexity: 8083.18,
    crossEntropy: 8.998,
    top8Next: [
      { rank: 1, tokenId: 892, text: " think", displaySymbol: "␣think", probability: 0.0723 },
      { rank: 2, tokenId: 1842, text: " love", displaySymbol: "␣love", probability: 0.0672 },
      { rank: 3, tokenId: 588, text: " like", displaySymbol: "␣like", probability: 0.0636 },
      { rank: 4, tokenId: 13, text: ".", displaySymbol: ".", probability: 0.0512 },
      { rank: 5, tokenId: 764, text: " can", displaySymbol: "␣can", probability: 0.0489 },
      { rank: 6, tokenId: 1037, text: " see", displaySymbol: "␣see", probability: 0.0381 },
      { rank: 7, tokenId: 284, text: " to", displaySymbol: "␣to", probability: 0.0321 },
      { rank: 8, tokenId: 290, text: " and", displaySymbol: "␣and", probability: 0.0298 }
    ]
  },
  "The capital of France is Paris": {
    tokens: [
      { index: 1, text: "The", displaySymbol: "The", tokenId: 464, probability: 0.009545, inverseProbability: 104.77, context: "[BOS]" },
      { index: 2, text: " capital", displaySymbol: "␣capital", tokenId: 3139, probability: 0.000242, inverseProbability: 4127.27, context: "[BOS] The" },
      { index: 3, text: " of", displaySymbol: "␣of", tokenId: 286, probability: 0.152018, inverseProbability: 6.58, context: "[BOS] The ␣capital" },
      { index: 4, text: " France", displaySymbol: "␣France", tokenId: 4881, probability: 0.002886, inverseProbability: 346.49, context: "[BOS] The ␣capital ␣of" },
      { index: 5, text: " is", displaySymbol: "␣is", tokenId: 318, probability: 0.074391, inverseProbability: 13.44, context: "[BOS] The ␣capital ␣of ␣France" },
      { index: 6, text: " Paris", displaySymbol: "␣Paris", tokenId: 6342, probability: 0.000080, inverseProbability: 12438.20, context: "[BOS] The ␣capital ␣of ␣France ␣is" }
    ],
    perplexity: 234.15,
    crossEntropy: 5.456,
    top8Next: [
      { rank: 1, tokenId: 11, text: ",", displaySymbol: ",", probability: 0.3621 },
      { rank: 2, tokenId: 13, text: ".", displaySymbol: ".", probability: 0.2814 },
      { rank: 3, tokenId: 290, text: " and", displaySymbol: "␣and", probability: 0.0789 },
      { rank: 4, tokenId: 326, text: " that", displaySymbol: "␣that", probability: 0.0245 },
      { rank: 5, tokenId: 351, text: " with", displaySymbol: "␣with", probability: 0.0187 },
      { rank: 6, tokenId: 284, text: " to", displaySymbol: "␣to", probability: 0.0162 },
      { rank: 7, tokenId: 373, text: " but", displaySymbol: "␣but", probability: 0.0141 },
      { rank: 8, tokenId: 547, text: " as", displaySymbol: "␣as", probability: 0.0118 }
    ]
  }
};

export const CONTEXT_EXPERIMENTS = [
  {
    id: "exp-order",
    label: "Grammar Order",
    a: "I love Python",
    b: "Python love I",
    insight: "Identical vocabulary, scrambled syntax — observe the order-of-magnitude perplexity spike."
  },
  {
    id: "exp-context",
    label: "Rich vs. Bare Context",
    a: "The capital of France is Paris",
    b: "Paris",
    insight: "Compare predicting a fact in context versus an isolated city name."
  },
  {
    id: "exp-code",
    label: "Commented Code",
    a: "# Python script\nprint('hello world')",
    b: "print('hello world')",
    insight: "Test whether a language-hint comment reduces surprise on subsequent code tokens."
  }
];

export const RACE_PROMPTS = [
  {
    id: "race-news",
    label: "Science Headline",
    prefix: "Astronomers discovered a new exoplanet orbiting"
  },
  {
    id: "race-code",
    label: "Python Function",
    prefix: "def fibonacci(n):\n    if n <="
  },
  {
    id: "race-story",
    label: "Story Opening",
    prefix: "Late at night, the observatory doors slowly"
  },
  {
    id: "race-fact",
    label: "Geography Fact",
    prefix: "The largest ocean on Earth is the"
  }
];

export const SANDBOX_PRESETS = {
  "preset-a": {
    tokens: [
      { text: "I", prob: 0.5 },
      { text: "love", prob: 0.8 },
      { text: "Python", prob: 0.2 }
    ],
    note: "Inverses are 2, 1.25, and 5. Multiplying yields 12.5, and the cube root (12.5)¹/³ ≈ <strong>2.32</strong>."
  },
  "preset-b": {
    tokens: Array.from({ length: 8 }, (_, i) => ({ text: `tok_${i + 1}`, prob: 0.125 })),
    note: "Every token has P = 0.125 = 1/8, so every inverse is 8. The 8th root of 8⁸ is <strong>exactly 8.00</strong> — proving perplexity equals the effective branching factor."
  },
  "preset-c": {
    tokens: [
      { text: "The", prob: 0.9 },
      { text: "sky", prob: 0.9 },
      { text: "is", prob: 0.9 },
      { text: "quantum-banana", prob: 0.01 }
    ],
    note: "Three tokens have 90% confidence, but one surprise at P = 0.01 injects an inverse penalty of <strong>100</strong>, pulling the entire geometric mean sharply upward."
  },
  "preset-d": {
    tokens: [
      { text: "North", prob: 0.25 },
      { text: "South", prob: 0.25 },
      { text: "East", prob: 0.25 },
      { text: "West", prob: 0.25 }
    ],
    note: "Four equally likely directions (P = 0.25 each) yield Perplexity = (4⁴)¹/⁴ = <strong>4.00</strong>."
  }
};

export const QUIZ_DATA = [
  {
    q: "What does a lower perplexity score indicate about a language model's predictions?",
    options: [
      "The model assigned higher probabilities to the actual tokens and was less surprised.",
      "The model was confused and guessed randomly across the vocabulary.",
      "The model required more GPU memory to evaluate the sequence.",
      "The input sequence contains ungrammatical syntax."
    ],
    correct: 0,
    explanation: "Perplexity measures average surprise. Lower perplexity means higher probability mass fell on the actual observed tokens."
  },
  {
    q: "If every token in an 8-token sentence has probability P = 0.125 (1/8), what is the perplexity?",
    options: ["0.125", "1.00", "8.00", "64.00"],
    correct: 2,
    explanation: "Each inverse is 1 / (1/8) = 8. The 8th root of 8⁸ equals exactly 8.00."
  },
  {
    q: "In an autoregressive causal model like DistilGPT-2, how is token w₃ predicted?",
    options: [
      "Conditioned on future tokens w₄ and w₅.",
      "Conditioned strictly on preceding context: P(w₃ | w₁, w₂).",
      "Independently of every other token in the sequence.",
      "By averaging probabilities across the entire training corpus."
    ],
    correct: 1,
    explanation: "Causal language models are left-to-right (autoregressive): each token is predicted solely from preceding tokens."
  },
  {
    q: "Why does GPT-2 evaluation prepend token ID 50256 (<|endoftext|>) before the first word?",
    options: [
      "It acts as an authentication key for the ONNX weights.",
      "It provides a starting-context position so the first token receives a conditioned prediction.",
      "It switches the model from classification mode to translation mode.",
      "It disables softmax normalization on the first layer."
    ],
    correct: 1,
    explanation: "Token 50256 (<|endoftext|>) serves as the BOS (beginning-of-sequence) context convention so position 0 can predict token 1."
  },
  {
    q: "What happens if a model assigns a tiny probability (e.g. 0.0001) to a single unexpected token?",
    options: [
      "Perplexity barely moves because the remaining tokens cancel it out.",
      "Perplexity spikes sharply because 1/P becomes 10,000 for that token.",
      "Perplexity drops to zero.",
      "The tokenizer strips the rare token before scoring."
    ],
    correct: 1,
    explanation: "Because perplexity takes the geometric mean of inverse probabilities (or exponential of average NLL), one extreme surprise heavily lifts the score."
  },
  {
    q: "Why do production ML pipelines compute perplexity via exp(-1/N ∑ ln P) instead of multiplying probabilities directly?",
    options: [
      "To prevent floating-point underflow when multiplying many small probabilities.",
      "Because browser JavaScript cannot multiply decimals smaller than 0.01.",
      "Logarithms skip the need to run the neural network forward pass.",
      "It changes the mathematical value to be 10× smaller."
    ],
    correct: 0,
    explanation: "Multiplying hundreds of probabilities below 1 rapidly underflows to 0.0 in IEEE-754 floats. Summing log-probabilities avoids underflow."
  },
  {
    q: "Why is 'Python love I' (PPL ≈ 8,083) dramatically higher than 'I love Python' (PPL ≈ 502)?",
    options: [
      "Capitalizing 'Python' triggers an out-of-vocabulary penalty.",
      "Broken English word order makes each subsequent token far less probable given prior context.",
      "DistilGPT-2 penalizes sentences under 5 words.",
      "The word 'Python' is blocked in position 1."
    ],
    correct: 1,
    explanation: "Unnatural word order clashes with learned statistical grammar, causing tiny conditional probabilities and a massive perplexity spike."
  },
  {
    q: "Can you fairly compare perplexity numbers between two LLMs that use different tokenizers?",
    options: [
      "Yes — perplexity is completely invariant to how text is split.",
      "No — perplexity depends on token granularity and is only comparable on identical data and tokenization.",
      "Only if both models have the same parameter count.",
      "Yes, as long as both models use temperature T = 1.0."
    ],
    correct: 1,
    explanation: "Fewer, longer tokens vs. more, shorter tokens distribute uncertainty differently per step, so cross-tokenizer perplexity comparisons are invalid."
  }
];

export const CONCEPTS_DATA = [
  {
    tag: "CORE ENGINE",
    title: "Autoregressive Language Model",
    formula: "P(w₁…wₙ) = ∏ P(wᵢ | w₁…wᵢ₋₁)",
    body: "A statistical sequence model that predicts one token at a time from left to right. At every position, it outputs raw logits across its entire vocabulary and normalizes them into probabilities via softmax."
  },
  {
    tag: "SUBWORD UNIT",
    title: "Token & BPE Tokenization",
    formula: "'␣Python' → ID 11361",
    body: "Models don't read raw words or characters alone—they use Byte-Pair Encoding (BPE) to split text into frequent subword chunks. Leading spaces attach to the following token (shown here as ␣)."
  },
  {
    tag: "DECODING",
    title: "Temperature, Top-K & Top-P",
    formula: "pᵢ = exp(zᵢ / T) / ∑ exp(zⱼ / T)",
    body: "Temperature scales logits before softmax; Top-K keeps only the K highest-probability tokens; Top-P (nucleus) keeps the smallest set whose cumulative probability reaches P."
  },
  {
    tag: "INFORMATION THEORY",
    title: "Surprise (Bits & Inverses)",
    formula: "I(w) = −log₂ P(w) bits",
    body: "A rare event carries more information (surprise) than an obvious one. Inverse probability 1/P measures effective choices, while −log₂ P expresses that same surprise in binary bits."
  },
  {
    tag: "OBJECTIVE",
    title: "Cross-Entropy Loss",
    formula: "H = −(1/N) ∑ ln P(wᵢ | ctx)",
    body: "Cross-entropy is the average negative log-likelihood per token during training. Minimizing cross-entropy forces the model to push probability mass onto the true next token."
  },
  {
    tag: "EVALUATION",
    title: "Perplexity (PPL)",
    formula: "PPL = exp(H) = (∏ 1/Pᵢ)^(1/N)",
    body: "Perplexity exponentiates cross-entropy back out of log-space into an intuitive branching factor: how many equally likely choices the model effectively felt it was choosing between at each step."
  }
];
