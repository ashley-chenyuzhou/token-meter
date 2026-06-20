"""Token Meter feature extractor — Python side.
MUST stay byte-for-byte equivalent in behavior to featurize() in model.js.
Tested for parity in tmdev/parity_test."""
import re, math

KHASH = 128        # hashed bag-of-words buckets
D = 23 + KHASH     # 23 engineered features + word-hash features

_UNITS = [
    ("tokens",     r"(\d{1,6})[\s-]*tokens?",                                                   16, 1.0),
    ("words",      r"(\d{1,5})[\s-]*words?",                                                     17, 1.0),
    ("sentences",  r"(\d{1,4})[\s-]*sentences?",                                                 18, 1.0),
    ("paragraphs", r"(\d{1,4})[\s-]*paragraphs?",                                                19, 1.0),
    ("pages",      r"(\d{1,3})[\s-]*pages?",                                                     20, 1.0),
    ("chars",      r"(\d{1,7})[\s-]*char(?:acter)?s?",                                           21, 1.0),
    ("count",      r"(\d{1,4})[\s-]*(?:bullets?|points?|items?|steps?|examples?|reasons?|ways?|tips?|ideas?)", 22, 1.0),
]

def estimate_tokens(text):
    if not text:
        return 0
    chars = len(text)
    words = len(re.findall(r"\S+", text))
    return max(1, round((chars / 4 + words * 1.33) / 2))

def featurize(text, input_tokens):
    t = (text or "").lower()
    def has(rx):
        return 1.0 if re.search(rx, t) else 0.0
    f = [0.0] * D
    f[0]  = math.log1p(max(0, input_tokens or 0)) / 7
    f[1]  = has(r"\b(summ?ari|tl;?dr|condens|shorten|abridg)")
    f[2]  = has(r"\b(explain|what is|what are|how does|how do|teach|describ|elaborat|why)\b")
    f[3]  = has(r"\b(writ|creat|build|implement|generat|develop).{0,40}(code|function|script|class|component|api|endpoint|sql|regex|program)")
    f[4]  = has(r"\b(debug|fix|refactor|optimi[sz]|review).{0,30}(code|bug|function|error|script)")
    f[5]  = has(r"\b(list|enumerat|brainstorm|suggest|recommend|give me)")
    f[6]  = has(r"\b(compar|contrast|versus|\bvs\.?|pros and cons|difference)")
    f[7]  = has(r"\b(rewrit|rephras|revis|proofread|paraphras|polish|\bedit)")
    f[8]  = has(r"(\?\s*$)|^(is|are|was|were|do|does|did|can|could|should|will|would|has|have)\b")
    f[9]  = has(r"\b(essay|article|blog|report|story|poem|write about|draft|novel|script)\b")
    f[10] = has(r"\b(opinion|thoughts|your view|do you (think|believe))")
    f[11] = has(r"\b(detailed|in depth|in-depth|comprehensive|thorough|step by step|step-by-step|elaborate|fully|at length|deep dive)\b")
    f[12] = has(r"\b(brief|briefly|short|concise|quick|tldr|tl;dr|in a sentence|one line|in short|succinct)\b")
    f[13] = has(r"```|\bcode\b")
    # ── Quantity (number + unit). Model learns the per-unit multiplier itself. ──
    for _name, rx, slot, _scale in _UNITS:
        mt = re.search(rx, t)
        if mt:
            n = int(mt.group(1))
            for i in range(1, 14):            # explicit size overrides task-type verbosity
                f[i] = 0.0
            f[14] = 1.0                       # has explicit quantity
            f[15] = math.log1p(n) / 9         # the requested number (normalized)
            f[slot] = 1.0                     # which unit
            break
    # ── Hashed bag-of-words (skipped when an explicit size is given, so the
    #    stated size dominates): the net learns which words predict length ──
    if f[14] == 0:
        for w in re.findall(r"[a-z0-9]+", t)[:120]:
            h = 0
            for ch in w:
                h = (h * 31 + ord(ch)) % 4294967296
            f[23 + (h % KHASH)] = 1.0
    return f
