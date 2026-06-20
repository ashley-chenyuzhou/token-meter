// Token Meter — content script for claude.ai
// Design rule for this rebuild: INJECT THE WIDGET FIRST, count tokens second.
// Every token/prediction operation is wrapped defensively so that a failure in
// counting can never prevent the widget from rendering. There is no Web Worker
// (the previous build's worker was likely killed by claude.ai's CSP, which took
// the whole content script down before injection ever ran).

(function () {
  'use strict';

  if (window.__tokenMeterLoaded) return;     // guard against double-injection
  window.__tokenMeterLoaded = true;

  // ─── Settings ────────────────────────────────────────────────────────────
  let settings = { enabled: true, contextLimit: 'auto' };
  try {
    chrome.storage?.local.get(['settings', 'widgetPos', 'modelWeights', 'history'], (d) => {
      if (d && d.settings) settings = Object.assign(settings, d.settings);
      if (d && d.widgetPos) dragPos = d.widgetPos;
      if (window.TMModel) {
        if (d && d.modelWeights) window.TMModel.importWeights(d.modelWeights);
        else if (d && d.history) retrainModel(d.history);
      }
      if (d && d.history) updateLearnStats(d.history);
      const w0 = document.getElementById('tm-widget');
      if (w0) applySavedPosition(w0);
      render();
    });
    chrome.storage?.onChanged.addListener((changes, area) => {
      if (area === 'local' && changes.settings) {
        settings = Object.assign(settings, changes.settings.newValue || {});
        if (!settings.enabled) removeWidget(); else ensureWidget();
        render();
      }
    });
  } catch (e) { /* storage unavailable — fall back to defaults */ }

  // ─── State ───────────────────────────────────────────────────────────────
  let inputTokens = 0;
  let conversationTokens = 0;
  let apiTokens = 0;             // full-conversation tokens from claude.ai's own data (0 if unavailable)
  let domTokens = 0;             // tokens summed from rendered messages (live updates / fallback)
  let lastApiFetch = 0;          // throttle timestamp for the conversation fetch
  let lastAsstUuid = null;       // uuid of the most recent assistant reply (to detect a NEW one)
  let lastAssistantTokens = 0;   // length of the most recent reply (for "do that again")
  let pred = { low: 0, mid: 0, high: 0 };
  let learnStats = { n: 0, cov: null };   // how much it has learned from this user

  // ─── Token estimate (dependency-free) ───────────────────────────────────
  // Not an exact BPE count, but a stable blend of char- and word-based
  // estimates. Typically within ~10–15% of real counts for prose. Exact
  // tokenization can be layered on later without touching the rest of the UI.
  function estimateTokens(text) {
    if (!text) return 0;
    const chars = text.length;
    const words = (text.match(/\S+/g) || []).length;
    const byChars = chars / 4;
    const byWords = words * 1.33;
    return Math.max(1, Math.round((byChars + byWords) / 2));
  }

  // ─── Output predictor ────────────────────────────────────────────────────
  // Returns {low, mid, high} predicted output tokens. Transparent and rule-based:
  // explicit length constraints win; otherwise a task-type table; otherwise a
  // length-based fallback.
  function predictOutput(text, inCount) {
    if (!text || inCount === 0) return { low: 0, mid: 0, high: 0 };
    const t = text.toLowerCase();

    const wrap = (mid, lo, hi, floor) => ({
      low: Math.max(floor, Math.round(mid * lo)),
      mid: Math.max(floor, Math.round(mid)),
      high: Math.max(floor, Math.round(mid * hi)),
    });

    // 1. Explicit size constraints — strongest signal. Handles "300 tokens",
    //    "500-word", "2 pages", "3 sentences"… (space or hyphen). Token requests
    //    are noisy — models can't count their own tokens — so the band is wide.
    let m;
    if ((m = t.match(/(\d{1,6})[\s-]*tokens?/)))          return wrap(+m[1] * 1.15, 0.8,  1.7,  10);
    if ((m = t.match(/(\d{1,5})[\s-]*words?/)))           return wrap(+m[1] * 1.35, 0.85, 1.15, 20);
    if ((m = t.match(/(\d{1,4})[\s-]*sentences?/)))       return wrap(+m[1] * 20,   0.8,  1.2,  20);
    if ((m = t.match(/(\d{1,4})[\s-]*paragraphs?/)))      return wrap(+m[1] * 80,   0.8,  1.25, 40);
    if ((m = t.match(/(\d{1,3})[\s-]*pages?/)))           return wrap(+m[1] * 650,  0.7,  1.4,  120);
    if ((m = t.match(/(\d{1,7})[\s-]*char(?:acter)?s?/))) return wrap(+m[1] / 4 * 1.05, 0.85, 1.3, 10);
    if ((m = t.match(/(\d{1,4})[\s-]*(bullets?|points?|items?|steps?|examples?|reasons?|ways?|tips?|ideas?)/)))
      return wrap(+m[1] * 25, 0.8, 1.3, 40);

    // 2. Task-type table: [regex, midRatio, lowRatio, highRatio, floor]
    // Stems use a leading \b only (no trailing \b) so inflections match:
    // "summar" -> summarize/summary, "function" -> functions, etc.
    const tasks = [
      [/\b(summ?ari|tl;?dr|condens|shorten|abridg)/,                        0.25, 0.6, 1.6, 60],
      [/\btranslat/,                                                        1.0,  0.8, 1.2, 40],
      [/\b(yes or no|true or false)\b/,                                     0.15, 0.4, 2.0, 10],
      [/^(is|are|was|were|do|does|did|can|could|should|will|would|has|have)\b.{0,80}\?$/, 0.2, 0.4, 2.5, 15],
      [/\b(writ|creat|build|implement|generat|develop).{0,40}(code|function|script|class|component|api|endpoint|sql|regex|program)/, 2.5, 0.6, 1.8, 120],
      [/\b(debug|fix|refactor|optimi[sz]|review).{0,30}(code|bug|function|error|script)/, 1.3, 0.5, 2.0, 80],
      [/\b(explain|what is|what are|how does|how do|teach|describ|elaborat|\bwhy\b)/, 1.6, 0.5, 2.2, 100],
      [/\b(rewrit|rephras|revis|proofread|paraphras|polish|\bedit)/,        1.0,  0.7, 1.4, 50],
      [/\b(list|enumerat|brainstorm|suggest|recommend|give me)/,            1.3,  0.5, 2.2, 80],
      [/\b(compar|contrast|versus|\bvs\.?|pros and cons|difference)/,       1.4,  0.6, 2.0, 120],
      [/\b(essay|article|blog|report|story|poem|write about|draft)/,        3.0,  0.5, 1.8, 250],
      [/\b(opinion|thoughts|your view|do you (think|believe))/,             0.9,  0.4, 1.8, 70],
    ];
    for (const [re, mid, lo, hi, floor] of tasks) {
      if (re.test(t)) {
        const base = Math.max(inCount, floor);
        return wrap(base * mid, lo, hi, floor);
      }
    }

    // 3. Length-based fallback
    if (inCount < 30)  return wrap(inCount * 1.5, 0.4, 3.0, 30);
    if (inCount < 200) return wrap(inCount * 1.0, 0.4, 2.5, 40);
    return wrap(inCount * 0.5, 0.25, 1.4, 60);
  }

  // ─── Learned predictor glue (the net lives in model.js) ──────────────────
  // The net predicts absolute length from features that INCLUDE the requested
  // quantity, so it learns "300 tokens / 5 bullets / 2 pages → length" itself
  // rather than from hand-written rules. predictOutput() below is only the
  // cold-start fallback used until the model is ready.
  // Re-fit from the user's logged (features → actual) history, then save.
  function retrainModel(history) {
    try {
      if (!window.TMModel) return;
      const usable = (history || []).filter(h => h.f && h.actual && h.actual >= 1);
      const N = usable.length;
      const samples = usable.map((h, i) => ({
        f: h.f, y: Math.log(h.actual),
        w: Math.max(0.2, Math.pow(0.94, N - 1 - i)),   // recent replies weighted more (tracks current behavior)
      }));
      window.TMModel.fit(samples);
      chrome.storage?.local.set({ modelWeights: window.TMModel.exportWeights() });
      updateLearnStats(history);
    } catch (e) { /* ignore */ }
  }

  // How much has it learned from this user? (count of replies + recent accuracy)
  function updateLearnStats(history) {
    try {
      const h = history || [];
      learnStats.n = h.filter(x => x.f && x.actual && x.actual >= 1).length;
      // "in range" = how often the actual reply landed inside the predicted range
      const recent = h.filter(x => x.actual && x.low != null && x.high != null).slice(-30);
      if (recent.length >= 3) {
        const inRange = recent.filter(x => x.actual >= x.low && x.actual <= x.high).length;
        learnStats.cov = Math.round(100 * inRange / recent.length);
      }
      render();
    } catch (e) { /* ignore */ }
  }

  // ─── Composer detection ──────────────────────────────────────────────────
  function findComposer() {
    const sel =
      document.querySelector('div[contenteditable="true"].ProseMirror') ||
      document.querySelector('[data-testid="chat-input"]') ||
      document.querySelector('[contenteditable="true"][role="textbox"]') ||
      document.querySelector('[role="textbox"][contenteditable="true"]') ||
      document.querySelector('div[contenteditable="true"]') ||
      document.querySelector('textarea');
    return sel || null;
  }

  function isComposer(el) {
    if (!el || el.nodeType !== 1) return false;
    return el.matches?.('div[contenteditable="true"], [role="textbox"], textarea') &&
           !!el.closest('form, [class*="composer"], main, body');
  }

  function readComposerText(el) {
    if (!el) return '';
    return ('value' in el && el.value != null) ? el.value : (el.innerText || '');
  }

  // ─── Conversation token estimate (debounced) ─────────────────────────────
  let convoTimer = null;
  let seenMsgs = {};   // content-keyed token counts — stable under claude.ai's list virtualization
  function scheduleConvoCount() {
    clearTimeout(convoTimer);
    convoTimer = setTimeout(() => {
      try {
        fetchConversationTokens();   // authoritative full count from claude.ai's own data (no scrolling needed)
        // claude.ai only renders messages near the viewport, so summing the DOM
        // directly changes as you scroll. Instead, accumulate every message we've
        // seen (keyed by role + a stable text prefix) and sum that — monotonic,
        // converges to the true total, counts both your prompts and Claude's replies.
        const nodes = document.querySelectorAll(
          '[data-testid="user-message"], [data-testid="assistant-message"], .font-claude-message'
        );
        nodes.forEach(n => {
          const txt = (n.innerText || '').trim();
          if (!txt) return;
          const role = n.matches('[data-testid="user-message"]') ? 'u' : 'a';
          const key = role + '|' + txt.slice(0, 80).replace(/\s+/g, ' ');
          seenMsgs[key] = estimateTokens(txt);   // overwrite (handles a streaming reply growing)
        });
        let total = 0; for (const k in seenMsgs) total += seenMsgs[k];
        domTokens = total;
        conversationTokens = Math.max(apiTokens, domTokens);   // never show less than what's on screen
        const asst = document.querySelectorAll('[data-testid="assistant-message"], .font-claude-message');
        const lastAsst = asst[asst.length - 1];
        if (lastAsst) lastAssistantTokens = estimateTokens(lastAsst.innerText || '');
        detectModel();
        render();
      } catch (e) { /* leave previous value */ }
    }, 900);
  }

  // ─── Authoritative conversation count via claude.ai's own data ────────────
  // The DOM only contains messages near the viewport (virtualized list), so it
  // undercounts when you open an existing chat until you scroll up. Instead we
  // read the full conversation from the same endpoint the app uses (same-origin,
  // your own session, read-only). Counts every message — your prompts AND
  // Claude's replies. Falls back to the DOM sum if anything here fails.
  async function fetchConversationTokens(force) {
    try {
      const m = location.pathname.match(/\/chat\/([0-9a-f-]{36})/i);
      if (!m) { apiTokens = 0; return; }                          // new/empty chat → DOM handles it
      if (!force && Date.now() - lastApiFetch < 1200) return;     // throttle (re-fetches as the chat changes)
      lastApiFetch = Date.now();
      const conv = m[1];
      let org = (document.cookie.match(/lastActiveOrg=([0-9a-f-]{36})/i) || [])[1];
      if (!org) {
        const orgs = await fetch('/api/organizations', { credentials: 'include' }).then(r => r.ok ? r.json() : null).catch(() => null);
        if (Array.isArray(orgs) && orgs.length) org = orgs[0].uuid || orgs[0].id;
      }
      if (!org) return;
      const url = `/api/organizations/${org}/chat_conversations/${conv}?tree=True&rendering_mode=messages`;
      const data = await fetch(url, { credentials: 'include' }).then(r => r.ok ? r.json() : null).catch(() => null);
      const msgs = data && (data.chat_messages || data.messages);
      if (!Array.isArray(msgs) || !msgs.length) return;
      let total = 0, lastAsst = null;
      for (const mm of msgs) {
        let t = (typeof mm.text === 'string' && mm.text) ? mm.text : '';
        if (!t && Array.isArray(mm.content)) for (const c of mm.content) { if (c && typeof c.text === 'string') t += c.text + '\n'; }
        const tok = estimateTokens(t);
        total += tok;                                              // count BOTH your prompts and Claude's replies
        if ((mm.sender || mm.role) === 'assistant') lastAsst = { uuid: mm.uuid, tok };
      }
      apiTokens = total;
      conversationTokens = Math.max(apiTokens, domTokens);
      if (lastAsst && lastAsst.tok > 0) {
        lastAssistantTokens = lastAsst.tok;
        // A new assistant reply appeared → learn from it (pair it with the prompt we predicted on).
        if (lastAsst.uuid && lastAsst.uuid !== lastAsstUuid) {
          lastAsstUuid = lastAsst.uuid;
          if (pendingPrediction) { recordLearning(pendingPrediction, lastAsst.tok); pendingPrediction = null; }
        }
      }
      render();
    } catch (e) { /* fall back to the DOM sum */ }
  }

  // ─── Model + context-window detection ────────────────────────────────────
  // claude.ai shows the active model in the composer's model picker. We read it
  // and map to the chat context window: Opus 4.6/4.7/4.8 and Sonnet 4.6 = 500K,
  // everything else = 200K. (Source: Anthropic Help Center, June 2026.)
  let detected = null;   // { label, limit }

  function detectModel() {
    try {
      let best = '';
      const els = document.querySelectorAll(
        'button, [role="button"], [data-testid*="model" i], [aria-label*="model" i]'
      );
      for (const el of els) {
        const t = (el.textContent || '').replace(/\s+/g, ' ').trim();
        if (t && t.length <= 32 && /(opus|sonnet|haiku)\s*\d/i.test(t)) { best = t; break; }
      }
      if (!best) return;
      const limit = (/opus\s*4\.[678]/i.test(best) || /sonnet\s*4\.6/i.test(best)) ? 500000 : 200000;
      const m = best.match(/(opus|sonnet|haiku)\s*[\d.]+/i);
      detected = { label: m ? m[0] : best, limit };
    } catch (e) { /* ignore */ }
  }

  // Effective limit: manual override (a number) wins; otherwise the detected
  // model's window; otherwise a conservative 200K.
  function effectiveLimit() {
    if (typeof settings.contextLimit === 'number') return settings.contextLimit;
    return (detected && detected.limit) || 200000;
  }

  // ─── Widget ──────────────────────────────────────────────────────────────
  function buildWidget() {
    const el = document.createElement('div');
    el.id = 'tm-widget';
    el.innerHTML = `
      <div class="tm-head">
        <span class="tm-grip" title="Drag to move">⠿</span>
        <span class="tm-title">Token Meter</span>
        <span class="tm-badge" id="tm-badge"></span>
      </div>
      <div class="tm-row">
        <div class="tm-stat">
          <span class="tm-label">Message</span>
          <span class="tm-num" id="tm-input">0</span><span class="tm-unit">tok</span>
        </div>
        <div class="tm-sep"></div>
        <div class="tm-stat">
          <span class="tm-label">Reply</span>
          <span class="tm-num" id="tm-output">—</span><span class="tm-unit" id="tm-range"></span>
        </div>
      </div>
      <div class="tm-track" title="Context window usage">
        <div class="tm-fill tm-fill-used" id="tm-used"></div>
        <div class="tm-fill tm-fill-pred" id="tm-pred"></div>
      </div>
      <div class="tm-foot"><span id="tm-summary">type to estimate</span><span class="tm-est" id="tm-model">est.</span></div>
      <div class="tm-learn" id="tm-learn"></div>
    `;
    return el;
  }

  function ensureWidget() {
    if (!settings.enabled) return;
    if (document.getElementById('tm-widget')) return;   // already present
    if (!findComposer()) return;   // only show on pages that have a chat input

    // Float as a fixed, draggable card appended to <body>, so it can never
    // overlap claude.ai's composer internals. Defaults to the bottom-right;
    // the user can drag it anywhere and the position is remembered.
    const widget = buildWidget();
    document.body.appendChild(widget);
    applySavedPosition(widget);
    makeDraggable(widget);
    detectModel();
    scheduleConvoCount();
    render();
  }

  function removeWidget() {
    document.getElementById('tm-widget')?.remove();
  }

  // ─── Dragging (fixed-position floating card) ─────────────────────────────
  let dragPos = null;   // {left, top} persisted to storage
  let drag = null;      // active drag session

  function applySavedPosition(w) {
    if (!dragPos) return;
    w.style.left = dragPos.left + 'px';
    w.style.top = dragPos.top + 'px';
    w.style.right = 'auto';
    w.style.bottom = 'auto';
    clampIntoView(w);
  }

  function clampIntoView(w) {
    const r = w.getBoundingClientRect();
    const maxL = window.innerWidth - r.width - 6;
    const maxT = window.innerHeight - r.height - 6;
    w.style.left = Math.min(Math.max(6, r.left), Math.max(6, maxL)) + 'px';
    w.style.top = Math.min(Math.max(6, r.top), Math.max(6, maxT)) + 'px';
    w.style.right = 'auto';
    w.style.bottom = 'auto';
  }

  function makeDraggable(w) {
    w.addEventListener('mousedown', (e) => {
      const r = w.getBoundingClientRect();
      drag = { w, ox: r.left, oy: r.top, sx: e.clientX, sy: e.clientY };
      w.style.left = r.left + 'px';
      w.style.top = r.top + 'px';
      w.style.right = 'auto';
      w.style.bottom = 'auto';
      w.classList.add('tm-dragging');
      e.preventDefault();
    });
  }

  document.addEventListener('mousemove', (e) => {
    if (!drag) return;
    drag.w.style.left = (drag.ox + e.clientX - drag.sx) + 'px';
    drag.w.style.top  = (drag.oy + e.clientY - drag.sy) + 'px';
  });
  document.addEventListener('mouseup', () => {
    if (!drag) return;
    const w = drag.w; drag = null;
    w.classList.remove('tm-dragging');
    clampIntoView(w);
    const r = w.getBoundingClientRect();
    dragPos = { left: Math.round(r.left), top: Math.round(r.top) };
    try { chrome.storage?.local.set({ widgetPos: dragPos }); } catch (e) { /* ignore */ }
  });
  window.addEventListener('resize', () => {
    const w = document.getElementById('tm-widget');
    if (w && dragPos) clampIntoView(w);
  });

  function render() {
    const w = document.getElementById('tm-widget');
    if (!w) return;
    const $ = (id) => w.querySelector('#' + id);

    const limit  = effectiveLimit();
    const used   = conversationTokens + inputTokens;
    const usedPct = Math.min(100, (used / limit) * 100);
    const predPct = Math.min(100 - usedPct, (pred.mid / limit) * 100);
    const remaining = Math.max(0, limit - used - pred.mid);

    $('tm-input').textContent  = inputTokens.toLocaleString();
    $('tm-output').textContent = pred.mid > 0 ? '≈' + pred.mid.toLocaleString() : '—';
    $('tm-range').textContent  = pred.mid > 0
      ? `${pred.low.toLocaleString()}–${pred.high.toLocaleString()}`
      : '';

    $('tm-used').style.width = usedPct.toFixed(1) + '%';
    $('tm-pred').style.width = predPct.toFixed(1) + '%';

    $('tm-summary').textContent = pred.mid > 0
      ? `${used.toLocaleString()} used · ${remaining.toLocaleString()} left of ${(limit/1000)|0}k`
      : `${used.toLocaleString()} of ${(limit/1000)|0}k used`;

    const modelEl = $('tm-model');
    if (modelEl) {
      const base = detected ? `${detected.label} · ${(limit/1000)|0}k` : `${(limit/1000)|0}k`;
      const M = window.TMModel;
      const tag = (!M || !M.ready) ? 'est' : (M.trained ? 'tuned' : 'model');
      modelEl.textContent = `${base} · ${tag}`;
    }

    const badge = $('tm-badge');
    const worstPct = (used + pred.high) / limit;
    badge.className = 'tm-badge';
    if (pred.mid <= 0) { badge.textContent = ''; }
    else if (worstPct > 0.95) { badge.textContent = '⚠ may not fit'; badge.classList.add('tm-danger'); }
    else if (worstPct > 0.75) { badge.textContent = '⚡ getting close'; badge.classList.add('tm-warn'); }
    else { badge.textContent = '✓ plenty of space'; badge.classList.add('tm-ok'); }

    // Learning indicator — shows it getting smarter from this user's replies
    const learnEl = $('tm-learn');
    if (learnEl) {
      if (learnStats.n <= 0) {
        learnEl.textContent = ''; learnEl.className = 'tm-learn';
      } else if (learnStats.n < 2) {
        learnEl.textContent = `learning — ${learnStats.n}/2 replies to start tuning to you`;
        learnEl.className = 'tm-learn';
      } else {
        learnEl.textContent = `✦ tuned on ${learnStats.n} of your replies`
          + (learnStats.cov != null ? ` · ${learnStats.cov}% in range` : '');
        learnEl.className = 'tm-learn tm-learn-on';
      }
    }
  }

  // ─── Live input handling (delegated so it survives re-renders) ───────────
  // Short, anaphoric prompts ("do that again", "another one") carry no length
  // signal in themselves — the answer is in the previous turn. Detect them so we
  // can inherit the last reply's length instead of mis-reading these few words.
  function isContinuation(text) {
    const t = (text || '').trim().toLowerCase();
    if (!t || t.length > 64) return false;
    return /\b(do (that|it|this) again|try again|again|another( one)?|same( thing| again)?|repeat( that| it)?|one more|do more|continue|keep going|go on|more of (that|the same)|redo)\b/.test(t);
  }

  function handleInput(el) {
    try {
      const text = readComposerText(el);
      inputTokens = estimateTokens(text);
      let features = null;
      if (isContinuation(text) && lastAssistantTokens > 0) {
        // Inherit the previous reply's length. Not logged for training — it's a
        // property of the conversation, not of these few words.
        const n = lastAssistantTokens;
        pred = { low: Math.round(n * 0.7), mid: n, high: Math.round(n * 1.4) };
      } else if (inputTokens > 0 && window.TMModel && window.TMModel.ready) {
        features = window.TMModel.featurize(text, inputTokens);
        pred = window.TMModel.predict(features);            // learned model
      } else {
        pred = predictOutput(text, inputTokens);            // cold-start rule fallback
        if (inputTokens > 0 && window.TMModel) features = window.TMModel.featurize(text, inputTokens);
      }
      // Snapshot the prediction. When the box goes from text → empty, that's a send,
      // so the snapshot becomes "pending" and gets paired with the reply for learning.
      if (inputTokens > 0) {
        lastNonEmptyPred = { input: inputTokens, mid: pred.mid, low: pred.low, high: pred.high, features };
      } else if (lastNonEmptyPred) {
        pendingPrediction = lastNonEmptyPred;
        lastNonEmptyPred = null;
      }
      render();
    } catch (e) { /* never let counting break the widget */ }
  }

  document.addEventListener('input', (e) => {
    if (isComposer(e.target)) handleInput(e.target);
  }, true);

  // ─── Adaptive learning: record actual reply length (from claude.ai's data) ──
  let pendingPrediction = null;   // prediction for a sent message, awaiting its reply
  let lastNonEmptyPred = null;    // most recent prediction made while the box had text
  function recordLearning(lp, actualTokens) {
    try {
      if (!lp || actualTokens < 1) return;
      const entry = { input: lp.input, predicted: lp.mid, low: lp.low, high: lp.high, actual: actualTokens, ts: Date.now(), f: lp.features || null };
      chrome.storage?.local.get(['history'], ({ history = [] }) => {
        history.push(entry);
        if (history.length > 200) history = history.slice(-200);
        chrome.storage?.local.set({ history });
        retrainModel(history);        // fine-tune from this new data point
        updateLearnStats(history);    // refresh the visible "tuned on N" indicator
      });
    } catch (e) { /* ignore */ }
  }

  // ─── Observe page for injection, convo changes, and reply completion ─────
  const mo = new MutationObserver(() => {
    ensureWidget();        // (re)inject if missing
    scheduleConvoCount();  // debounced — recounts + detects new replies via the API
  });
  mo.observe(document.documentElement, { childList: true, subtree: true });

  // Backup boot poll for the first injection
  const boot = setInterval(() => {
    ensureWidget();
    if (document.getElementById('tm-widget')) clearInterval(boot);
  }, 700);

  // Re-inject + reset on SPA navigation (new chat)
  let lastUrl = location.href;
  setInterval(() => {
    if (location.href !== lastUrl) {
      lastUrl = location.href;
      inputTokens = 0; conversationTokens = 0; pred = { low: 0, mid: 0, high: 0 };
      pendingPrediction = null; lastNonEmptyPred = null; seenMsgs = {};   // new conversation → fresh count
      apiTokens = 0; domTokens = 0; lastAsstUuid = null; lastApiFetch = 0;
      setTimeout(ensureWidget, 600);
    }
  }, 600);

  ensureWidget();
})();
