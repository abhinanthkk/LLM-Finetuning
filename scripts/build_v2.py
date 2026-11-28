"""Stage 2e: assemble the knowledge-injection dataset in data/qa_v2/.

  train.jsonl        every fact except the 'unseen' ones: original question + generated rewordings (same answer),
                     plus the OGMP document text in ~150-token passages (the model reads the source itself)
  val.jsonl          60 trained facts, each asked with a held-back rewording that is NOT in train
  test_seen.jsonl    70 trained facts, each asked with a hand-written rewording (data/qa_v2/test_rewordings.json)
  test_unseen.jsonl  20 facts that are in no training QA pair (only in the passage text), original wording
usage: python scripts/build_v2.py
"""
import json
import random
import re
from pathlib import Path

from transformers import AutoTokenizer

from split_qa import SYSTEM, to_row

ROOT = Path(__file__).resolve().parent.parent
V2 = ROOT / "data/qa_v2"
PASSAGE_TOKENS = 150


def words(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


def passages(chunks, tok):
    """Split each chunk into ~150-token pieces at line/sentence boundaries, as 'quote the guidance' examples."""
    rows = []
    for c in chunks:
        units = [u for line in c["text"].split("\n") for u in re.split(r"(?<=[.;])\s+(?=[A-Z(\[])", line) if u.strip()]
        pieces, cur = [], ""
        for u in units:
            if cur and len(tok(cur + " " + u)["input_ids"]) > PASSAGE_TOKENS:
                pieces.append(cur)
                cur = u
            else:
                cur = f"{cur} {u}".strip() if cur else u
        if cur:
            pieces.append(cur)
        title = re.sub(r"^Number \d+:\s*", "", c["doc_title"])
        for k, piece in enumerate(pieces, 1):
            rows.append({"id": f"{c['id']}-p{k}", "chunk_id": c["id"], "doc": c["doc"], "type": "passage",
                         "source": "document",
                         "prompt": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": f"Quote the OGMP {title} guidance, section "
                                                                f"\"{c['section']}\", part {k} of {len(pieces)}."}],
                         "completion": [{"role": "assistant", "content": piece}]})
    return rows


def main():
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    pairs = {p["id"]: p for p in map(json.loads, open(ROOT / "data/qa/qa_raw.jsonl", encoding="utf-8"))}
    para = {r["id"]: r["paraphrases"] for r in map(json.loads, open(ROOT / "data/qa/paraphrases.jsonl", encoding="utf-8"))}
    sel = json.load(open(V2 / "selection.json"))
    rewordings = json.load(open(V2 / "test_rewordings.json"))
    assert set(rewordings) == set(sel["test_seen"]), "every test_seen fact needs exactly one hand rewording"
    chunks = [json.loads(l) for l in open(ROOT / "data/chunks.jsonl", encoding="utf-8")]

    train, val = [], []
    for pid in sel["train_only"] + sel["test_seen"] + sel["val"]:
        p = pairs[pid]
        qs = [p["question"]] + para.get(pid, [])
        if pid in sel["val"]:
            if len(qs) < 2:
                continue                                   # no held-back wording available: train-only fact
            val.append(to_row(dict(p, question=qs[-1])))   # the last rewording is never trained on
            qs = qs[:-1]
        for k, q in enumerate(qs):
            train.append(to_row(dict(p, id=f"{pid}-w{k}", question=q)))
    n_qa = len(train)
    train += passages(chunks, tok)
    random.Random(0).shuffle(train)

    test_seen = [to_row(dict(pairs[pid], question=rewordings[pid])) for pid in sel["test_seen"]]
    test_unseen = [to_row(pairs[pid]) for pid in sel["unseen"]]

    # self-checks: nothing from the unseen facts in train QA; test/val wordings are not training wordings
    train_q = {r["prompt"][-1]["content"] for r in train}
    assert not any(r["id"].rsplit("-w", 1)[0] in sel["unseen"] for r in train), "unseen fact leaked into train"
    assert not any(r["prompt"][-1]["content"] in train_q for r in val + test_seen + test_unseen), "test wording in train"
    trained_wordings = {}
    for r in train:
        trained_wordings.setdefault(r["id"].rsplit("-w", 1)[0], []).append(words(r["prompt"][-1]["content"]))
    overlap = [max(len(words(r["prompt"][-1]["content"]) & w) / len(words(r["prompt"][-1]["content"]) | w)
                   for w in trained_wordings[r["id"]]) for r in test_seen]

    for name, rows in (("train", train), ("val", val), ("test_seen", test_seen), ("test_unseen", test_unseen)):
        with open(V2 / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"train {len(train)} ({n_qa} QA wordings of {len(sel['train_only']) + len(sel['test_seen']) + len(sel['val'])} "
          f"facts, {len(train) - n_qa} document passages) | val {len(val)} | test_seen {len(test_seen)} | "
          f"test_unseen {len(test_unseen)}")
    print(f"hand-written test wording vs closest trained wording of the same fact: mean word overlap "
          f"{sum(overlap) / len(overlap):.2f}, max {max(overlap):.2f}")


if __name__ == "__main__":
    main()
