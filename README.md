Token Meter for Claude

A browser extension that predicts how many tokens your message and Claude's reply will use, directly on claude.ai, before you hit send, so you can tell whether your next exchange fits inside the model's context window. The prediction comes from a small neural network that runs entirely in the browser and personalizes to your own usage.


Not affiliated with, endorsed by, or sponsored by Anthropic. "Claude" is a trademark of Anthropic, used here only to describe compatibility.


What it does


Live token count of your message as you type.
Predicted reply length shown as a calibrated range (via conformal prediction), not a single overconfident number.
A context-window usage bar and a fit / warning badge.
Learns from your own replies and gets more accurate the more you use it.
Fully local: no servers, no analytics, no data leaves your browser.


How it works

Token Meter treats response-length prediction as a black-box, client-side problem: it sees only the prompt text and, after the fact, the rendered reply. It never has access to the model's weights, activations, logits, or an API. Most prior work on length prediction is server-side and uses model internals; this project explores the harder client-only setting.


Features (151-dim): engineered signals (input length, task type, explicit size requests such as "300 tokens" or "2 pages", verbosity cues) plus a 128-dim hashed bag-of-words, so the network also learns directly from wording.
Base model: a two-hidden-layer MLP (151 -> 24 -> 8 -> 1) trained offline with backpropagation on roughly 31,000 real Claude conversations from LMSYS-Chat-1M, augmented with synthetic examples covering explicit-size and long-form prompts.
Calibrated intervals: split-conformal prediction produces distribution-free, guaranteed-coverage ranges instead of overconfident point estimates.
On-device personalization: after each reply, the extension compares its prediction to the actual length and updates a lightweight, recency-weighted correction, adapting to your model and your style. Everything stays in local browser storage.


Results (held-out test set, LMSYS Claude data)

ModelMedian APESpearman rhoGlobal mean79%0.02Linear (151 features)76%0.55MLP (this work)53%0.64

Calibrated intervals reach the target 80% coverage using split-conformal prediction.

A central finding: response length is heavy-tailed and partly irreducible. Across prompts that appear many times, about 9% of length variance is unpredictable in principle, implying a best-possible median APE of roughly 43%. The goal is therefore a well-calibrated range, not an exact count, and the MLP (53%) operates close to that floor.

Install (developer mode)


Download or clone this repository.
Go to chrome://extensions, enable Developer mode, click Load unpacked, and select the folder.
Open https://claude.ai.


(Or install from the Chrome Web Store once published.)

Repository structure


manifest.json, content.js, model.js, base_model.js, widget.css, popup.* : the extension.
training/ : the ML pipeline (feature extractor, base-model trainer, LMSYS data extractor, evaluation experiments).
store/ : privacy policy and Chrome Web Store listing copy.


Limitations


Token counts are estimates from a fast local approximation, not Claude's exact tokenizer.
The base model is trained on 2023-era Claude data and relies on online personalization to adapt to current models.
Reply length is inherently variable, so the extension reports a range; exact point prediction is fundamentally bounded (see the results section).


Privacy

All processing happens locally in your browser. No conversation data is sent to the developer or any third party. See store/PRIVACY.md.

License

MIT License. Copyright 2026 Ashley Zhou.
