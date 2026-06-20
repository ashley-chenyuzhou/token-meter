// Token Meter — learned predictor (neural net base + online correction)
//
// Two-stage learning, as designed with the user:
//   1. BASE: a tiny net (D→H→1) trained offline on a dataset by train_base.py
//      and shipped as base_model.js (window.__TM_BASE). Predicts ABSOLUTE log
//      output length from features that INCLUDE the requested quantity (number +
//      unit), so it learns "300 tokens / 5 bullets / 2 pages → length" itself
//      rather than from hand-written rules.
//   2. ONLINE: a LINEAR correction over the base, trained with backprop on the
//      user's own (prompt → actual length) pairs and L2-regularized toward zero.
//      Because it's pulled to zero, prompt types the user hasn't produced keep a
//      ~0 correction and fall back to the base — so personalizing on, say, long
//      "explain" replies can't inflate an explicit "300 tokens" request.
//
// The rule predictor (content.js) is only a cold-start fallback before the model
// is ready. Output length is high-variance, so the band is the residual spread.

(function (g) {
  'use strict';

  // ─── Features (MUST match featurize.py exactly) ─────────────────────────
  const KHASH = 128;
  const D = 23 + KHASH;
  const UNITS = [
    [/(\d{1,6})[\s-]*tokens?/, 16],
    [/(\d{1,5})[\s-]*words?/, 17],
    [/(\d{1,4})[\s-]*sentences?/, 18],
    [/(\d{1,4})[\s-]*paragraphs?/, 19],
    [/(\d{1,3})[\s-]*pages?/, 20],
    [/(\d{1,7})[\s-]*char(?:acter)?s?/, 21],
    [/(\d{1,4})[\s-]*(?:bullets?|points?|items?|steps?|examples?|reasons?|ways?|tips?|ideas?)/, 22],
  ];
  function featurize(text, inputTokens) {
    const t = (text || '').toLowerCase();
    const has = (re) => (re.test(t) ? 1 : 0);
    const f = new Array(D).fill(0);
    f[0]  = Math.log1p(Math.max(0, inputTokens || 0)) / 7;
    f[1]  = has(/\b(summ?ari|tl;?dr|condens|shorten|abridg)/);
    f[2]  = has(/\b(explain|what is|what are|how does|how do|teach|describ|elaborat|why)\b/);
    f[3]  = has(/\b(writ|creat|build|implement|generat|develop).{0,40}(code|function|script|class|component|api|endpoint|sql|regex|program)/);
    f[4]  = has(/\b(debug|fix|refactor|optimi[sz]|review).{0,30}(code|bug|function|error|script)/);
    f[5]  = has(/\b(list|enumerat|brainstorm|suggest|recommend|give me)/);
    f[6]  = has(/\b(compar|contrast|versus|\bvs\.?|pros and cons|difference)/);
    f[7]  = has(/\b(rewrit|rephras|revis|proofread|paraphras|polish|\bedit)/);
    f[8]  = has(/(\?\s*$)|^(is|are|was|were|do|does|did|can|could|should|will|would|has|have)\b/);
    f[9]  = has(/\b(essay|article|blog|report|story|poem|write about|draft|novel|script)\b/);
    f[10] = has(/\b(opinion|thoughts|your view|do you (think|believe))/);
    f[11] = has(/\b(detailed|in depth|in-depth|comprehensive|thorough|step by step|step-by-step|elaborate|fully|at length|deep dive)\b/);
    f[12] = has(/\b(brief|briefly|short|concise|quick|tldr|tl;dr|in a sentence|one line|in short|succinct)\b/);
    f[13] = has(/```|\bcode\b/);
    for (const [re, slot] of UNITS) {
      const m = t.match(re);
      if (m) {
        for (let i = 1; i < 14; i++) f[i] = 0;   // explicit size overrides task-type verbosity
        f[14] = 1; f[15] = Math.log1p(parseInt(m[1], 10)) / 9; f[slot] = 1;
        break;
      }
    }
    // ── Hashed bag-of-words (skipped when an explicit size is given): the net
    //    learns which words predict length ──
    const toks = f[14] === 0 ? t.match(/[a-z0-9]+/g) : null;
    if (toks) for (let n = 0; n < toks.length && n < 120; n++) {
      const w = toks[n]; let h = 0;
      for (let c = 0; c < w.length; c++) h = (h * 31 + w.charCodeAt(c)) % 4294967296;
      f[23 + (h % KHASH)] = 1;
    }
    return f;
  }

  // ─── Base net:  D → H1 (tanh) → H2 (tanh) → 1 (absolute log-tokens) ──────
  const H1 = 24, H2 = 8;
  const NWC = 23;   // online correction uses only the engineered features (stable)
  const OUT_MIN = Math.log(3), OUT_MAX = Math.log(190000);
  const SIG_MIN = 0.18, SIG_MAX = 1.1, SIG0 = 0.5;
  const DEFAULT_GCAL = 0.30;   // cold-start prior: current Claude ≈ 1.35× the 2023-trained base

  function smallInit() {
    let x = 777; const r = () => { x = (x * 1103515245 + 12345) & 0x7fffffff; return (x / 0x7fffffff - 0.5) * 0.2; };
    const mat = (rows, cols) => Array.from({ length: rows }, () => Array.from({ length: cols }, r));
    return { W1: mat(H1, D), b1: new Array(H1).fill(0), W2: mat(H2, H1),
             b2: new Array(H2).fill(0), W3: Array.from({ length: H2 }, r), b3: Math.log(150) };
  }
  function clone(w) {
    return { W1: w.W1.map(r => r.slice()), b1: w.b1.slice(), W2: w.W2.map(r => r.slice()),
             b2: w.b2.slice(), W3: w.W3.slice(), b3: w.b3 };
  }

  let BASE = smallInit();              // day-one weights (from base_model.js if present)
  let Wc = new Array(NWC).fill(0);     // online correction (engineered features only)
  let bc = 0;
  let gcal = DEFAULT_GCAL;   // global verbosity calibration vs current Claude (online)
  let sigma = SIG0;
  let qConf = null;          // split-conformal interval half-width (log space) from user residuals
  let hasBase = false;
  let trainedUser = false;

  function baseLog(f) {
    const A1 = new Array(H1);
    for (let j = 0; j < H1; j++) {
      let a = BASE.b1[j]; const row = BASE.W1[j];
      for (let i = 0; i < D; i++) a += row[i] * f[i];
      A1[j] = Math.tanh(a);
    }
    let s = BASE.b3;
    for (let k = 0; k < H2; k++) {
      let a = BASE.b2[k]; const row = BASE.W2[k];
      for (let j = 0; j < H1; j++) a += row[j] * A1[j];
      s += BASE.W3[k] * Math.tanh(a);
    }
    return s;
  }
  function corr(f) { let c = bc; for (let i = 0; i < NWC; i++) c += Wc[i] * f[i]; return c; }
  function logMean(f) { return baseLog(f) + corr(f) + (f[14] ? 0 : gcal); }

  // Online: fit a linear correction (engineered features only) on samples [{f, y}].
  // L2 pulls Wc/bc toward 0, so prompt types the user hasn't produced stay at the
  // base — personalizing on long replies can't inflate an explicit "300 tokens".
  function fit(samples) {
    if (!samples || samples.length < 2) { Wc = new Array(NWC).fill(0); bc = 0; gcal = DEFAULT_GCAL; qConf = null; trainedUser = false; return; }
    // 1. Global calibration FIRST: median lift from the base on non-size prompts.
    //    Captures "this user's Claude runs longer than the base" and generalizes
    //    to prompt types not yet seen. (Explicit-size prompts set their own length.)
    const r0 = [];
    for (const s of samples) { if (!s.f[14]) r0.push(s.y - baseLog(s.f)); }
    gcal = r0.length ? Math.max(-1.5, Math.min(1.5, r0.sort((a, b) => a - b)[r0.length >> 1])) : DEFAULT_GCAL;
    // 2. Per-feature correction fits what's LEFT after base + gcal (per-task
    //    deviations). l2w << l2b so a verbose topic can't inflate other prompts.
    const lr = 0.05, epochs = 250, l2w = 0.015, l2b = 0.2;
    const wc = new Array(NWC).fill(0); let b = 0;
    const idx = samples.map((_, i) => i);
    for (let e = 0; e < epochs; e++) {
      let seed = 4242 + e * 13;
      for (let i = idx.length - 1; i > 0; i--) { seed = (seed * 1103515245 + 12345) & 0x7fffffff; const jj = seed % (i + 1); const tmp = idx[i]; idx[i] = idx[jj]; idx[jj] = tmp; }
      for (const m of idx) {
        const s = samples[m]; const f = s.f; const sw = (s.w == null) ? 1 : s.w;  // recency weight
        const g = f[14] ? 0 : gcal;
        let c = b; for (let i = 0; i < NWC; i++) c += wc[i] * f[i];
        let err = (baseLog(f) + g + c) - s.y;
        if (err > 4) err = 4; if (err < -4) err = -4;
        for (let i = 0; i < NWC; i++) wc[i] -= lr * (sw * err * f[i] + l2w * wc[i]);
        b -= lr * (sw * err + l2b * b);
      }
    }
    Wc = wc; bc = b;
    let se = 0; const absr = [];
    for (const s of samples) { const r = logMean(s.f) - s.y; se += r * r; absr.push(Math.abs(r)); }
    sigma = Math.max(0.25, Math.min(SIG_MAX, Math.sqrt(se / samples.length)));
    // Split-conformal half-width: empirical 80th percentile of |residual| once we
    // have enough of the user's own data → distribution-free coverage, no Gaussian assumption.
    if (absr.length >= 8) { absr.sort((a, b) => a - b); qConf = absr[Math.min(absr.length - 1, Math.ceil((absr.length + 1) * 0.80) - 1)]; }
    else qConf = null;
    trainedUser = true;
  }

  function predict(f, z) {
    z = (z == null) ? 1.2816 : z;   // 80% Gaussian fallback (used until conformal has data)
    const out = Math.max(OUT_MIN, Math.min(OUT_MAX, logMean(f)));
    const mid = Math.exp(out);
    const w = (qConf != null) ? qConf : z * sigma;   // conformal half-width once enough user data
    return {
      mid: Math.round(mid),
      low: Math.round(mid * Math.exp(-w)),
      high: Math.round(mid * Math.exp(w)),
      sigma,
    };
  }

  function validShape(w) {
    return w && w.W1 && w.W1.length === H1 && w.W1[0] && w.W1[0].length === D &&
           w.b1 && w.b1.length === H1 && w.W2 && w.W2.length === H2 && w.W2[0] && w.W2[0].length === H1 &&
           w.b2 && w.b2.length === H2 && w.W3 && w.W3.length === H2 && typeof w.b3 === 'number';
  }
  function importBase(blob) {
    try {
      if (blob && blob.D === D && blob.H1 === H1 && blob.H2 === H2 && validShape(blob.W)) {
        BASE = clone(blob.W);
        sigma = Math.max(SIG_MIN, Math.min(SIG_MAX, blob.sigma || SIG0));
        hasBase = true;
      }
    } catch (e) { /* keep default */ }
  }
  function exportWeights() { return { Wc, bc, gcal, sigma, qConf, trainedUser, v: 7 }; }
  function importWeights(obj) {
    try {
      if (obj && Array.isArray(obj.Wc) && obj.Wc.length === NWC) {
        Wc = obj.Wc.slice(); bc = obj.bc || 0;
        gcal = (typeof obj.gcal === 'number') ? obj.gcal : DEFAULT_GCAL;
        qConf = (typeof obj.qConf === 'number') ? obj.qConf : null;
        sigma = Math.max(SIG_MIN, Math.min(SIG_MAX, obj.sigma || sigma));
        trainedUser = !!obj.trainedUser;
      }
    } catch (e) { /* keep current */ }
  }

  if (g.__TM_BASE) importBase(g.__TM_BASE);   // load shipped base, if present

  g.TMModel = {
    D, featurize, fit, predict, exportWeights, importWeights, importBase,
    get ready() { return hasBase || trainedUser; },
    get trained() { return trainedUser; },
    get hasBase() { return hasBase; },
    get sigma() { return sigma; },
  };
})(typeof window !== 'undefined' ? window : globalThis);
