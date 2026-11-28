"""Helper for writing QA pairs by hand (or by an assistant in chat) into make_qa.py's cache.
  python scripts/claude_batch.py show N        print the next N chunks that still need LLM pairs
  python scripts/claude_batch.py save FILE     FILE = {"<chunk_id>": [pairs...]}; checks each pair with
                                               make_qa.check_pair and writes passing chunks to the cache
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import make_qa as m

chunks = {c["id"]: c for c in map(json.loads, open(m.CHUNKS, encoding="utf-8"))}
todo = [c for c in chunks.values() if m.wants_llm(c) and not (m.CACHE / f"{c['id']}.json").exists()]

if sys.argv[1] == "show":
    for c in todo[:int(sys.argv[2])]:
        print(f"=== {c['id']} | write {m.n_pairs(c)} | {c['doc_title']} | {c['section']}\n{m.prose_of(c)}\n")
    print(f"({len(todo)} chunks left)")
elif sys.argv[1] == "save":
    data = json.load(open(sys.argv[2], encoding="utf-8"))
    for cid, pairs in data.items():
        bad = [(i, r) for i, p in enumerate(pairs) if (r := m.check_pair(p, chunks[cid]["text"]))]
        for i, r in bad:
            print(f"  {cid} pair {i}: {r} | evidence: {pairs[i]['evidence'][:90]}")
        m.CACHE.mkdir(parents=True, exist_ok=True)
        pairs = [dict(p, generator="claude-in-chat") for p in pairs]
        (m.CACHE / f"{cid}.json").write_text(json.dumps(pairs, ensure_ascii=False), encoding="utf-8")
        print(f"saved {cid}: {len(pairs) - len(bad)}/{len(pairs)} pass")
