#!/usr/bin/env python3
"""Generate a synthetic prompt->length dataset to (a) validate train_base.py and
(b) produce a sensible PLACEHOLDER base_model.js until a real dataset is provided.
Encodes realistic relationships incl. quantity->length so the net can learn them."""
import json, random, math, sys
random.seed(7)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
out = sys.argv[2] if len(sys.argv) > 2 else "synth.jsonl"

def lognorm(mu, sd):  # multiplicative noise
    return math.exp(random.gauss(math.log(mu), sd))

quantity_units = [
    ("{n} tokens", 1.12, 0.28), ("write {n} tokens", 1.12, 0.28), ("about {n} tokens", 1.1, 0.3),
    ("{n} words", 1.35, 0.16), ("a {n} word essay", 1.36, 0.16), ("{n}-word summary", 1.34, 0.18),
    ("{n} sentences", 21.0, 0.22), ("{n} paragraphs", 85.0, 0.24), ("{n} pages", 600.0, 0.3),
    ("{n} characters", 0.26, 0.2),
    ("{n} bullet points", 27.0, 0.3), ("list {n} ideas", 26.0, 0.32), ("{n} tips", 25.0, 0.3),
]
tasks = [
    ("summarize this article", 110, 0.5), ("tl;dr of the above", 90, 0.5),
    ("explain how {x} works", 360, 0.45), ("what is {x}", 220, 0.5), ("why does {x} happen", 320, 0.45),
    ("write a function to {x}", 330, 0.5), ("debug this code", 260, 0.55),
    ("write an essay about {x}", 820, 0.4), ("write a short story about {x}", 900, 0.45),
    ("is {x} true?", 45, 0.7), ("compare {x} and {y}", 360, 0.45),
    ("rewrite this paragraph", 150, 0.4), ("brainstorm ideas for {x}", 320, 0.5),
    ("what do you think about {x}", 240, 0.5), ("give me detailed steps for {x}", 520, 0.45),
]
topics = ["neural networks", "the economy", "photosynthesis", "rome", "react hooks", "climate", "chess", "coffee"]

def emit(prompt, tokens):
    return {"prompt": prompt, "output_tokens": int(max(1, round(tokens)))}

rows = []
for _ in range(N):
    if random.random() < 0.4:
        tmpl, factor, sd = random.choice(quantity_units)
        n = random.choice([1,2,3,5,8,10,20,50,100,150,200,300,500,800,1000,1500,2000])
        prompt = tmpl.format(n=n)
        rows.append(emit(prompt, n * factor * lognorm(1.0, sd)))
    else:
        tmpl, base, sd = random.choice(tasks)
        prompt = tmpl.format(x=random.choice(topics), y=random.choice(topics))
        # add some filler so input length varies
        if random.random() < 0.3:
            prompt += " " + " ".join(random.choice(topics) for _ in range(random.randint(3, 40)))
        rows.append(emit(prompt, base * lognorm(1.0, sd)))

with open(out, "w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")
print(f"wrote {len(rows)} rows to {out}")
