// Token Meter popup — reads/writes settings; content script reacts via storage.onChanged.

const enabledEl = document.getElementById('enabled');
const limitEl   = document.getElementById('limit');
const statEl    = document.getElementById('stat');

const DEFAULTS = { enabled: true, contextLimit: 'auto' };

function load() {
  chrome.storage.local.get(['settings', 'history'], ({ settings, history }) => {
    const s = Object.assign({}, DEFAULTS, settings || {});
    enabledEl.checked = s.enabled;
    limitEl.value = String(s.contextLimit);
    showStat(history || []);
  });
}

function save() {
  const v = limitEl.value;
  const settings = {
    enabled: enabledEl.checked,
    contextLimit: v === 'auto' ? 'auto' : (parseInt(v, 10) || 200000),
  };
  chrome.storage.local.set({ settings });
}

function showStat(history) {
  const tuned = history.filter(h => h.f && h.actual).length;
  const usable = history.filter(h => h.actual && h.low != null && h.high != null);
  if (usable.length < 2) {
    statEl.textContent = `Learning from your replies: ${usable.length}/2 to start tuning to you.`;
    return;
  }
  const recent = usable.slice(-50);
  const inRange = recent.filter(h => h.actual >= h.low && h.actual <= h.high).length;
  const cov = Math.round(100 * inRange / recent.length);
  statEl.innerHTML = `Tuned on <b>${tuned}</b> of your replies — the predicted range was right <b>${cov}%</b> of the time.`;
}

enabledEl.addEventListener('change', save);
limitEl.addEventListener('change', save);
load();
