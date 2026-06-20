#!/usr/bin/env python3
"""Experiments for the paper: baselines, point + interval (conformal) metrics,
and a predictability-ceiling estimate, on the LMSYS Claude response-length data.
Run: python3 experiments.py
"""
import json, math, os, numpy as np
from featurize import featurize, estimate_tokens, D
from train_base import train as train_mlp

HERE = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(0)

# ── Load data ────────────────────────────────────────────────────────────
rows = [json.loads(l) for l in open(os.path.join(HERE, "claude_data.jsonl"))]
prompts = [r["prompt"] for r in rows]
X = np.array([featurize(p, estimate_tokens(p)) for p in prompts], dtype=np.float64)
y = np.log(np.array([max(1, r["output_tokens"]) for r in rows], dtype=np.float64))
N = len(y)

# 70 / 15 / 15  train / calibration / test
perm = rng.permutation(N)
ntr, ncal = int(0.70 * N), int(0.15 * N)
tr, cal, te = perm[:ntr], perm[ntr:ntr + ncal], perm[ntr + ncal:]
Xtr, ytr, Xcal, ycal, Xte, yte = X[tr], y[tr], X[cal], y[cal], X[te], y[te]

def spearman(a, b):
    ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
    ra = ra - ra.mean(); rb = rb - rb.mean()
    return float((ra @ rb) / (np.sqrt((ra @ ra) * (rb @ rb)) + 1e-12))

def point_metrics(name, pred_log, ytrue_log):
    ape = np.abs(np.exp(pred_log) - np.exp(ytrue_log)) / np.exp(ytrue_log)
    print(f"  {name:<26} log-MAE {np.abs(pred_log-ytrue_log).mean():.3f} | "
          f"median APE {np.median(ape)*100:4.0f}% | Spearman {spearman(pred_log, ytrue_log):.3f}")

# ── Baselines + MLP (point prediction) ──────────────────────────────────
print("=== POINT PREDICTION (test set) ===")
# 1. global mean
mean_pred = np.full(len(yte), ytr.mean()); point_metrics("global mean", mean_pred, yte)
# 2. input-length only (ridge on just feature 0)
def ridge(Xa, ya, Xb, lam=1.0):
    A = Xa.T @ Xa + lam * np.eye(Xa.shape[1]); w = np.linalg.solve(A, Xa.T @ ya); return Xb @ w
len_pred = ridge(np.c_[np.ones(ntr), Xtr[:, :1]], ytr, np.c_[np.ones(len(te)), Xte[:, :1]])
point_metrics("input-length (ridge)", len_pred, yte)
# 3. linear ridge on all features
lin_pred = ridge(np.c_[np.ones(ntr), Xtr], ytr, np.c_[np.ones(len(te)), Xte])
point_metrics("linear (151 feats)", lin_pred, yte)
# 4. our MLP
W, sigma = train_mlp(Xtr, ytr, epochs=150)
W1, b1, W2, b2, W3, b3 = (np.array(W[k]) for k in ("W1", "b1", "W2", "b2", "W3", "b3"))
def mlp(Xa):
    A1 = np.tanh(Xa @ W1.T + b1); A2 = np.tanh(A1 @ W2.T + b2); return A2 @ W3 + b3
mlp_te = mlp(Xte); point_metrics("MLP 151->24->8->1 (ours)", mlp_te, yte)
# ablation: MLP, engineered features only (zero the hashed dims)
Xtr_e, Xcal_e, Xte_e = Xtr.copy(), Xcal.copy(), Xte.copy()
for A in (Xtr_e, Xcal_e, Xte_e): A[:, 23:] = 0
We, _ = train_mlp(Xtr_e, ytr, epochs=150)
W1e,b1e,W2e,b2e,W3e,b3e = (np.array(We[k]) for k in ("W1","b1","W2","b2","W3","b3"))
mlp_e_te = np.tanh(np.tanh(Xte_e@W1e.T+b1e)@W2e.T+b2e)@W3e+b3e
point_metrics("MLP, no word features", mlp_e_te, yte)

# ── Interval prediction + calibration (target 80% coverage) ─────────────
alpha = 0.20
def cov_width(name, lo, hi, ytrue):
    c = np.mean((ytrue >= lo) & (ytrue <= hi))
    print(f"  {name:<34} coverage {c*100:4.1f}% (target {int((1-alpha)*100)}%) | "
          f"median width x{math.exp(np.median(hi-lo)):.2f}")

print("\n=== INTERVAL PREDICTION (test set, target 80% coverage) ===")
# (a) Gaussian range from MLP residual std (the old approach)
z = 1.2816  # 80% two-sided
cov_width("Gaussian (MLP +/- z*sigma)", mlp_te - z*sigma, mlp_te + z*sigma, yte)

# (b) Split-conformal on MLP point predictions (distribution-free guarantee)
scores = np.abs(ycal - mlp(Xcal))
qhat = np.quantile(scores, min(1.0, (1 - alpha) * (len(cal) + 1) / len(cal)), method="higher")
cov_width("Split-conformal (MLP)", mlp_te - qhat, mlp_te + qhat, yte)

# (c) Conformalized Quantile Regression (Romano et al. 2019)
def pinball_fit(Xa, ya, tau, epochs=300, lr=0.3, lam=1e-3):
    Xb = np.c_[np.ones(len(Xa)), Xa]; w = np.zeros(Xb.shape[1]); n = len(Xb)
    for _ in range(epochs):
        r = ya - Xb @ w
        g = np.where(r > 0, -tau, (1 - tau))
        w -= lr * (Xb.T @ g / n + lam * w)
    return w
wlo = pinball_fit(Xtr, ytr, alpha/2); whi = pinball_fit(Xtr, ytr, 1 - alpha/2)
def qpred(w, Xa): return np.c_[np.ones(len(Xa)), Xa] @ w
qlo_cal, qhi_cal = qpred(wlo, Xcal), qpred(whi, Xcal)
E = np.maximum(qlo_cal - ycal, ycal - qhi_cal)
qc = np.quantile(E, min(1.0, (1 - alpha) * (len(cal) + 1) / len(cal)), method="higher")
cov_width("CQR (conformalized quantile reg)", qpred(wlo, Xte) - qc, qpred(whi, Xte) + qc, yte)

# ── Predictability ceiling: variance within identical prompts ───────────
print("\n=== PREDICTABILITY CEILING (duplicate prompts) ===")
from collections import defaultdict
groups = defaultdict(list)
for p, yi in zip(prompts, y): groups[p.strip().lower()].append(yi)
multi = [np.array(v) for v in groups.values() if len(v) >= 3]
within = np.concatenate([v - v.mean() for v in multi])
within_std = within.std()
total_std = y.std()
print(f"  {len(multi)} prompts seen >=3 times | within-prompt log-std {within_std:.3f} | "
      f"overall log-std {total_std:.3f}")
print(f"  -> ~{100*within_std**2/total_std**2:.0f}% of length variance is irreducible "
      f"(same prompt, different runs); best-possible median APE ~{100*(math.exp(0.674*within_std)-1):.0f}%")
