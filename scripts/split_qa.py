"""Stage 2b: data/qa/qa_raw.jsonl -> data/qa/{train,val,test}.jsonl (80/10/10), without leakage.

Pairs that ask about the same fact go to the same split. Two pairs are "the same fact" when they quote the
same evidence, or their questions share most of their words (Jaccard >= 0.6 on content words). Groups are
assigned per document, so every document appears in train, validation and test.

Output rows use the chat prompt-completion format the trainer expects, plus metadata for error analysis:
  {"id", "chunk_id", "doc", "type", "source",
   "prompt": [{"role": "system", ...}, {"role": "user", "content": question}],
   "completion": [{"role": "assistant", "content": answer}]}
usage: python scripts/split_qa.py
"""
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QA = ROOT / "data" / "qa"
SYSTEM = ("You are an assistant for oil and gas engineers. Answer questions about methane emissions "
          "using the OGMP (Oil & Gas Methane Partnership) technical guidance documents.")
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
STOP = set("a an the of for in on to and or is are what which how why does do with by from as at be "
           "that this according ogmp guidance give".split())


def words(s):
    return {w for w in re.findall(r"[a-z0-9]+(?:[.-][a-z0-9]+)*", s.lower()) if w not in STOP}


def groups_of(pairs, threshold=0.6):
    """Union-find over pairs: same evidence, or near-identical questions, are one group."""
    parent = list(range(len(pairs)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_evidence = {}
    for i, p in enumerate(pairs):
        e = re.sub(r"\s+", " ", p["evidence"].lower()).strip()
        if e in by_evidence:
            parent[find(i)] = find(by_evidence[e])
        by_evidence.setdefault(e, i)
    ws = [words(p["question"]) for p in pairs]
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            if pairs[i]["source"] == pairs[j]["source"] == "template":
                continue          # template questions differ only in the row name: distinct facts, grouped by evidence
            if ws[i] and ws[j] and len(ws[i] & ws[j]) >= threshold * len(ws[i] | ws[j]):
                parent[find(j)] = find(i)
    groups = defaultdict(list)
    for i in range(len(pairs)):
        groups[find(i)].append(i)
    return list(groups.values())


def assign(groups, pairs, seed=42):
    """Per document, shuffle its groups and fill val and test first (to ~10% of pairs each), rest to train."""
    rng = random.Random(seed)
    split_of = {}
    by_doc = defaultdict(list)
    for g in groups:
        by_doc[Counter(pairs[i]["doc"] for i in g).most_common(1)[0][0]].append(g)
    for doc in sorted(by_doc):
        gs = by_doc[doc]
        rng.shuffle(gs)
        total = sum(len(g) for g in gs)
        filled = Counter()
        for g in gs:
            target = next((s for s in ("test", "val") if filled[s] + len(g) <= max(1, round(SPLITS[s] * total))
                           and len(g) <= 3), "train")        # big groups (many rewordings) stay in train
            filled[target] += len(g)
            for i in g:
                split_of[i] = target
    return split_of


def to_row(p):
    return {"id": p["id"], "chunk_id": p["chunk_id"], "doc": p["doc"], "type": p["type"], "source": p["source"],
            "prompt": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": p["question"]}],
            "completion": [{"role": "assistant", "content": p["answer"]}]}


def main():
    pairs = [json.loads(l) for l in open(QA / "qa_raw.jsonl", encoding="utf-8")]
    groups = groups_of(pairs)
    split_of = assign(groups, pairs)

    # self-check: no group crosses splits, and no test question has a near-twin in train
    for g in groups:
        assert len({split_of[i] for i in g}) == 1, "a group was split"
    llm_train = [words(pairs[i]["question"]) for i in split_of if split_of[i] == "train" and pairs[i]["source"] == "llm"]
    worst = max((len(words(pairs[i]["question"]) & w) / len(words(pairs[i]["question"]) | w)
                 for i in split_of if split_of[i] == "test" and pairs[i]["source"] == "llm" for w in llm_train if w),
                default=0)

    for name in SPLITS:
        rows = [to_row(pairs[i]) for i in range(len(pairs)) if split_of[i] == name]
        with open(QA / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name:5s} {len(rows):4d} pairs | by source {dict(Counter(r['source'] for r in rows))} | "
              f"docs {len({r['doc'] for r in rows})}")
    print(f"{len(pairs)} pairs in {len(groups)} fact groups; largest group {max(map(len, groups))}")
    print(f"most similar test-vs-train LLM question pair: {worst:.2f} word overlap (groups use >= 0.60)")


if __name__ == "__main__":
    main()
