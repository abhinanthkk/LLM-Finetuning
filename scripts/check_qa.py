"""Audit every QA pair: data/qa/qa_raw.jsonl against data/chunks.jsonl, and the split files.
usage: python scripts/check_qa.py      prints every problem found, grouped by check
"""
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import make_qa as m

ROOT = m.ROOT
chunks = {c["id"]: c for c in map(json.loads, open(ROOT / "data/chunks.jsonl", encoding="utf-8"))}
pairs = [json.loads(l) for l in open(ROOT / "data/qa/qa_raw.jsonl", encoding="utf-8")]
problems = defaultdict(list)

ids = Counter(p["id"] for p in pairs)
for p in pairs:
    pid, q, a, e = p["id"], p["question"], p["answer"], p["evidence"]
    c = chunks.get(p["chunk_id"])
    if not c:
        problems["unknown chunk"].append(pid); continue
    if ids[pid] > 1:
        problems["duplicate id"].append(pid)
    if r := m.check_pair(p, c["text"]):
        problems[f"core check: {r}"].append(pid)
    if p["doc"] != c["doc"]:
        problems["doc mismatch"].append(pid)
    if not q.rstrip().endswith("?"):
        problems["question has no '?'"].append(f"{pid}: {q}")
    if re.search(r"\b(the|this) (passage|text|section|excerpt|chunk)\b|\babove\b|\bbelow\b", q, re.I):
        problems["question refers to the passage"].append(f"{pid}: {q}")
    if re.search(r"\b(the|this) (passage|text|excerpt|chunk)\b|\bas mentioned\b", a, re.I):
        problems["answer refers to the passage"].append(f"{pid}: {a[:100]}")
    if len(a.split()) < 3 or len(q.split()) < 4:
        problems["too short"].append(f"{pid}: Q={q} | A={a}")
    if re.search(r"\b(Ibid|http|www\.|doi)\b", q + a, re.I):
        problems["citation/URL content"].append(f"{pid}: {q}")
    # a unit attached to a number in the evidence should travel with it into the answer
    for num, unit in re.findall(r"(\d[\d,.]*)\s*(scfh|scmh|scf|scm|Mcf|Mscf|psig|percent|%)\b", e):
        if num in a and not re.search(re.escape(num) + r"\s*\(?\s*" + unit, a) and unit not in a:
            problems["number without its unit"].append(f"{pid}: {num} {unit} | A={a[:110]}")
    # yes/no questions must be answered yes/no
    if re.match(r"(Is|Are|Does|Do|Can|Should|Must)\b", q) and not re.match(r"(Yes|No)\b", a):
        problems["yes/no question not answered yes/no"].append(f"{pid}: {q} | A={a[:80]}")

# template answers: every value must equal the table row it came from
for p in pairs:
    if p["source"] == "template":
        row = m.parse_row(p["evidence"])
        vals = [v for _, v in row[2] if m.is_value(v)]
        if not all(v in p["answer"] for v in vals):
            problems["template value mismatch"].append(p["id"])

# near-duplicate questions inside one document (same fact asked twice)
words = lambda s: set(re.findall(r"[a-z0-9]+", s.lower())) - {"the", "a", "of", "for", "what", "is", "in", "to", "and", "how"}
by_doc = defaultdict(list)
for p in pairs:
    if p["source"] == "llm":
        by_doc[p["doc"]].append(p)
for doc, ps in by_doc.items():
    for i in range(len(ps)):
        for j in range(i + 1, len(ps)):
            wi, wj = words(ps[i]["question"]), words(ps[j]["question"])
            if wi and len(wi & wj) / len(wi | wj) >= 0.7:
                problems["near-duplicate questions"].append(f"{ps[i]['id']} ~ {ps[j]['id']}: {ps[i]['question']} || {ps[j]['question']}")

print(f"{len(pairs)} pairs checked")
for k, v in sorted(problems.items(), key=lambda kv: -len(kv[1])):
    print(f"\n[{k}] {len(v)}")
    for x in v[:40]:
        print("   ", x)
if not problems:
    print("no problems found")
