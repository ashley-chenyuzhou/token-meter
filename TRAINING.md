# Training the base model

The extension ships with `base_model.js` — the day-one model. The in-browser learner fine-tunes it on your own usage, but a base trained on **real Claude conversations** gives much better predictions before it has seen any of your data.

The base currently in the extension was trained on *synthetic* data (placeholder). This folder lets you replace it with one trained on a real dataset.

## What you provide

A single file, `.jsonl` or `.csv`, where each row is one user turn and Claude's reply. Any of these shapes works:

**JSONL** (one JSON object per line):
```
{"prompt": "explain how RSA works", "response": "RSA is a public-key… (full reply)"}
{"prompt": "summarize this", "output_tokens": 240}
```

**CSV** (with a header row):
```
prompt,response
"explain how RSA works","RSA is a public-key…"
```

- `prompt` = the user's message. (`input`/`instruction` also accepted.)
- Either `response` (the full reply text — its length is measured) **or** `output_tokens`/`completion_tokens` if you already have counts. (`output`/`completion`/`answer` also accepted.)
- More rows = better. A few thousand is good; tens of thousands is great.

## Where to get Claude conversations

- **LMSYS-Chat-1M** (`lmsys/lmsys-chat-1m` on HuggingFace) — tags each conversation by model; filter to the `claude-*` ones. Gated: you accept terms on the dataset page and use a HuggingFace account/token.
- **WildChat** — large, but GPT outputs, so lengths differ from Claude. Usable, less ideal.
- **Your own exported chats** — anything you can shape into the format above.

> I couldn't pull these inside this tool (HuggingFace is network-blocked here), which is why you run this step where you have normal internet.

## Run it

```bash
pip install numpy
python3 train_base.py --data your_dataset.jsonl --out base_model.js
```

It prints held-out accuracy, then writes `base_model.js`. Copy that file into the
extension folder (next to `model.js`), reload the extension, and refresh claude.ai.

`featurize.py` here is kept **byte-for-byte equivalent** to the feature extractor in
`model.js` — that parity is what makes the trained weights valid in the browser. If
you change features in one, change both.

(`generate_synth.py` regenerates the synthetic placeholder, for reference.)
