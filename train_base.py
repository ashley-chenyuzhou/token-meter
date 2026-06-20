#!/usr/bin/env python3
"""Token Meter — base model trainer.

Trains the tiny net on a dataset of (prompt -> reply length) and exports
base_model.js, which the extension loads as its day-one starting point. The
in-browser learner then keeps fine-tuning from there on the user's own usage.

USAGE:
    python3 train_base.py --data your_dataset.jsonl --out base_model.js

ACCEPTED INPUT (any one of these; .jsonl or .csv):
    JSONL, one object per line:
        {"prompt": "...user message...", "response": "...claude reply..."}
        {"prompt": "...", "output_tokens": 812}          # if you already have lengths
    CSV with a header row containing columns:
        prompt,response            OR            prompt,output_tokens

The prompt should be the user's message; the response (or output_tokens) is
what Claude replied. Only the *length* of the reply is used as the target.
"""
import argparse, json, csv, math, sys, re
import numpy as np
from featurize import featurize, estimate_tokens, D

H1 = 24   # first hidden layer
H2 = 8    # second hidden layer

def load_rows(path):
    rows = []
    if path.endswith(".jsonl") or path.endswith(".json"):
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    pass
    elif path.endswith(".csv") or path.endswith(".tsv"):
        delim = "\t" if path.endswith(".tsv") else ","
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh, delimiter=delim):
                rows.append(r)
    else:
        sys.exit("Unsupported file type: use .jsonl or .csv")
    return rows

def to_sample(row):
    """Return (features, log_output_tokens) or None if unusable."""
    prompt = row.get("prompt") or row.get("input") or row.get("instruction") or ""
    if not isinstance(prompt, str) or not prompt.strip():
        return None
    out_tok = row.get("output_tokens") or row.get("completion_tokens") or row.get("n_tokens")
    if out_tok is None:
        resp = row.get("response") or row.get("output") or row.get("completion") or row.get("answer") or ""
        if not isinstance(resp, str) or not resp.strip():
            return None
        out_tok = estimate_tokens(resp)
    try:
        out_tok = float(out_tok)
    except Exception:
        return None
    if out_tok < 1 or out_tok > 200000:
        return None
    f = featurize(prompt, estimate_tokens(prompt))
    return np.array(f, dtype=np.float64), math.log(out_tok)

def _fwd(P, X):
    A1 = np.tanh(X @ P["W1"].T + P["b1"])
    A2 = np.tanh(A1 @ P["W2"].T + P["b2"])
    out = A2 @ P["W3"] + P["b3"]
    return A1, A2, out

def train(X, y, epochs=300, lr=0.008, l2=0.002, seed=0):
    rng = np.random.default_rng(seed)
    P = dict(
        W1=rng.normal(0, 0.12, (H1, D)), b1=np.zeros(H1),
        W2=rng.normal(0, 0.15, (H2, H1)), b2=np.zeros(H2),
        W3=rng.normal(0, 0.15, H2), b3=float(np.mean(y)),   # start at mean log-length
    )
    mom = {k: np.zeros_like(np.asarray(v, dtype=float)) for k, v in P.items()}
    vel = {k: np.zeros_like(np.asarray(v, dtype=float)) for k, v in P.items()}
    bm, bv, t = 0.9, 0.999, 0
    N = len(X)
    for ep in range(epochs):
        idx = rng.permutation(N)
        for s in range(0, N, 128):
            bi = idx[s:s+128]; xb, yb = X[bi], y[bi]; nb = len(bi)
            A1, A2, out = _fwd(P, xb)
            d_out = (out - yb) / nb                       # dL/dout for 0.5*MSE
            gW3 = A2.T @ d_out + l2 * P["W3"]; gb3 = float(np.sum(d_out))
            dA2 = np.outer(d_out, P["W3"]) * (1 - A2**2)
            gW2 = dA2.T @ A1 + l2 * P["W2"]; gb2 = np.sum(dA2, axis=0)
            dA1 = (dA2 @ P["W2"]) * (1 - A1**2)
            gW1 = dA1.T @ xb + l2 * P["W1"]; gb1 = np.sum(dA1, axis=0)
            grads = dict(W1=gW1, b1=gb1, W2=gW2, b2=gb2, W3=gW3, b3=np.array(gb3))
            t += 1
            for k in P:
                g = grads[k]
                mom[k] = bm * mom[k] + (1 - bm) * g
                vel[k] = bv * vel[k] + (1 - bv) * (g * g)
                step = lr * (mom[k] / (1 - bm**t)) / (np.sqrt(vel[k] / (1 - bv**t)) + 1e-8)
                if k == "b3": P["b3"] -= float(step)
                else: P[k] -= step
    _, _, out = _fwd(P, X)
    sigma = max(0.18, min(1.1, float(np.sqrt(np.mean((out - y) ** 2)))))
    W = dict(W1=P["W1"].tolist(), b1=P["b1"].tolist(), W2=P["W2"].tolist(),
             b2=P["b2"].tolist(), W3=P["W3"].tolist(), b3=float(P["b3"]))
    return W, sigma

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="base_model.js")
    ap.add_argument("--epochs", type=int, default=400)
    a = ap.parse_args()

    rows = load_rows(a.data)
    samples = [s for s in (to_sample(r) for r in rows) if s is not None]
    if len(samples) < 20:
        sys.exit(f"Only {len(samples)} usable rows — need at least ~20 (ideally thousands).")
    X = np.array([s[0] for s in samples]); y = np.array([s[1] for s in samples])
    # 85/15 split for an honest metric
    n = len(X); cut = int(n * 0.85)
    perm = np.random.default_rng(0).permutation(n)
    tr, te = perm[:cut], perm[cut:]
    W, sigma = train(X[tr], y[tr], epochs=a.epochs)

    def predict(f):
        A1 = np.tanh(np.array(f) @ np.array(W["W1"]).T + np.array(W["b1"]))
        A2 = np.tanh(A1 @ np.array(W["W2"]).T + np.array(W["b2"]))
        return math.exp(A2 @ np.array(W["W3"]) + W["b3"])
    # held-out metrics
    med_err = np.median([abs(predict(X[i]) - math.exp(y[i])) / math.exp(y[i]) for i in te])
    cover = np.mean([abs(math.log(predict(X[i])) - y[i]) <= 1.04 * sigma for i in te])
    print(f"trained on {len(tr)} rows | held-out median error {med_err*100:.0f}% | coverage {cover*100:.0f}% | sigma {sigma:.2f}")

    blob = dict(W=W, sigma=sigma, D=D, H1=H1, H2=H2, v=5, n=len(X))
    with open(a.out, "w") as fh:
        fh.write("// Auto-generated base model for Token Meter. Drop into the extension folder.\n")
        fh.write("window.__TM_BASE = " + json.dumps(blob) + ";\n")
    print(f"wrote {a.out}")

if __name__ == "__main__":
    main()
