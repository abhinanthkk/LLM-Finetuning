"""Stage 2d: choose which facts are tested, for the knowledge-injection design (data/qa_v2/selection.json).

  test_seen   70 facts that stay in training; tested with a NEW wording of the question (written by hand)
  val         60 facts that stay in training; validated with a held-back generated rewording
  unseen      20 facts removed from training entirely; tested with the original question (the honest limit)
Facts = groups from split_qa.groups_of (same evidence or near-identical question), so a fact is never split.
Stratified by document, fixed seed. usage: python scripts/select_v2.py
"""
import json
import random
from collections import defaultdict
from pathlib import Path

from split_qa import groups_of

ROOT = Path(__file__).resolve().parent.parent
N = {"test_seen": 70, "val": 60, "unseen": 20}


def main():
    pairs = [json.loads(l) for l in open(ROOT / "data/qa/qa_raw.jsonl", encoding="utf-8")]
    groups = groups_of(pairs)
    rng = random.Random(7)
    by_doc = defaultdict(list)
    for g in groups:
        by_doc[pairs[g[0]]["doc"]].append(g)
    for gs in by_doc.values():
        rng.shuffle(gs)
    # round-robin over documents so every document contributes in proportion to its size
    order, docs = [], sorted(by_doc, key=lambda d: -len(by_doc[d]))
    while any(by_doc.values()):
        for d in docs:
            if by_doc[d]:
                order.append(by_doc[d].pop())
    sel, k = {}, 0
    for name, n in N.items():
        sel[name] = [pairs[i]["id"] for g in order[k:k + n] for i in g]
        k += n
    sel["train_only"] = [pairs[i]["id"] for g in order[k:] for i in g]
    (ROOT / "data/qa_v2").mkdir(exist_ok=True)
    json.dump(sel, open(ROOT / "data/qa_v2/selection.json", "w"), indent=1)
    for name, ids in sel.items():
        print(f"{name:11s} {len(ids):4d} pairs")
    by_id = {p["id"]: p for p in pairs}
    with open(ROOT / "data/qa_v2/test_seen_originals.jsonl", "w", encoding="utf-8") as f:
        for i in sel["test_seen"]:
            p = by_id[i]
            f.write(json.dumps({"id": i, "type": p["type"], "source": p["source"], "question": p["question"],
                                "answer": p["answer"]}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
