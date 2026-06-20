#!/usr/bin/env python3
"""Extract Claude (first user prompt -> first assistant reply length) pairs from
the LMSYS-Chat-1M parquet shards into claude_data.jsonl for train_base.py."""
import duckdb, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from featurize import estimate_tokens

HERE = os.path.dirname(os.path.abspath(__file__))
GLOB = os.path.join(HERE, "lmsys", "*.parquet")
OUT = os.path.join(HERE, "claude_data.jsonl")

def turn_field(turn, key, idx):
    if isinstance(turn, dict): return turn.get(key)
    try: return turn[idx]
    except Exception: return None

con = duckdb.connect()
rows = con.execute(
    f"SELECT conversation FROM read_parquet('{GLOB}') WHERE model ILIKE '%claude%'"
).fetchall()

n = 0
with open(OUT, "w") as fh:
    for (conv,) in rows:
        if not conv:
            continue
        prompt = reply = None
        for turn in conv:
            role = turn_field(turn, "role", 0)
            content = turn_field(turn, "content", 1)
            if not role or content is None:
                continue
            if prompt is None and role in ("user", "human"):
                prompt = content
            elif prompt is not None and role in ("assistant", "gpt", "bot", "model"):
                reply = content
                break
        if prompt and reply and prompt.strip():
            ot = estimate_tokens(reply)
            if 1 <= ot <= 20000:
                fh.write(json.dumps({"prompt": prompt[:2000], "output_tokens": ot}) + "\n")
                n += 1
print(f"wrote {n} Claude prompt->reply pairs to {OUT}")
