"""Stage 2c: up to three rewordings of every question in data/qa/qa_raw.jsonl -> data/qa/paraphrases.jsonl.

Paraphrase augmentation: a fine-tuned model absorbs a fact far better when it sees the fact asked several ways.
Only the QUESTION is reworded; the verified answer is reused unchanged, so no new facts (or errors) enter.

Generator: Qwen3-1.7B (local, pre-quantized 4-bit, thinking off). Every rewording is filtered:
  - meaning kept: embedding cosine with the original question >= 0.80 (BAAI/bge-small-en-v1.5)
  - actually reworded: shares < 85% of its words with the original
  - nothing lost: every number, table reference and acronym of the original is still in it, and the subject
    (content words the question shares with its answer, e.g. 'glycol dehydrators') is kept
  - a single question ending in '?', 4-60 words
usage: python scripts/paraphrase.py [--limit N]
"""
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
GEN = "unsloth/Qwen3-1.7B-unsloth-bnb-4bit"          # Qwen3-1.7B, pre-quantized to 4-bit (cached locally)
PROMPT = """Rewrite this question about oil and gas methane emissions in 5 different ways.
Rules: keep exactly the same meaning; keep every number, unit, table number and equipment name; always keep
the equipment or emission source the question is about.
Make each rewrite clearly different: use different words and a different sentence structure, and start each one
with a different word than the original (for example "Which", "In the OGMP guidance,", "For", "Could you explain",
"According to"). Each rewrite must be one question ending with "?".
Output exactly 5 lines, one question per line, nothing else.

Question: {q}"""


def must_keep(q):
    """Tokens a faithful rewording must still contain: numbers, 'Table 2.3', acronyms like VRU, OGI, CH4."""
    keep = set(re.findall(r"\d[\d,]*(?:\.\d+)?", q))
    keep |= set(re.findall(r"Table \d+\.\d+", q))
    keep |= {w for w in re.findall(r"\b[A-Z][A-Z0-9&]{1,}[a-z]?\b", q) if w not in {"OGMP"}}
    return keep


GENERIC = set("""what which when where how why does do the for and with from that this methane emissions emission
ogmp guidance partners partner oil gas according explain could would""".split())
stem = lambda w: re.sub(r"(ies|es|s|ing|ed)$", "", w)


def anchors(q, answer):
    """The question's subject: content words it shares with its answer ('storage tanks', 'glycol dehydrators')."""
    words = lambda s: {stem(w) for w in re.findall(r"[a-z][a-z-]{3,}", s.lower())} - {stem(g) for g in GENERIC}
    return words(q) & words(answer)


def valid(p, q, sim, answer=""):
    if not p.endswith("?") or p.count("?") > 1 or not 4 <= len(p.split()) <= 60:
        return "format"
    wp, wq = set(re.findall(r"\w+", p.lower())), set(re.findall(r"\w+", q.lower()))
    if len(wp & wq) >= 0.85 * len(wp | wq):          # embeddings rate any rewording ~0.95+, so compare words
        return "not reworded"
    if sim < 0.80:
        return "meaning drift"
    if any(k not in p for k in must_keep(q)):
        return "lost a number/term"
    a = anchors(q, answer)
    if a and len(a & {stem(w) for w in re.findall(r"[a-z][a-z-]{3,}", p.lower())}) < 0.85 * len(a):
        return "lost the subject"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--only-missing", action="store_true", help="retry questions that got no rewording, keep the rest")
    args = ap.parse_args()
    pairs = [json.loads(l) for l in open(ROOT / "data/qa/qa_raw.jsonl", encoding="utf-8")][:args.limit]
    previous = {}
    if args.only_missing:
        previous = {r["id"]: r for r in map(json.loads, open(ROOT / "data/qa/paraphrases.jsonl", encoding="utf-8"))}
        pairs = [p for p in pairs if not previous.get(p["id"], {}).get("paraphrases")]

    tok = AutoTokenizer.from_pretrained(GEN)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(GEN, device_map={"": 0}, dtype=torch.bfloat16).eval()
    raw = []
    for k in range(0, len(pairs), args.batch):
        msgs = [[{"role": "user", "content": PROMPT.format(q=p["question"])}] for p in pairs[k:k + args.batch]]
        enc = tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, padding=True,
                                      return_tensors="pt", return_dict=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=360, do_sample=True, temperature=0.9, top_p=0.9,
                                 pad_token_id=tok.pad_token_id)
        for g in out:
            text = tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True)
            lines = [re.sub(r"^\s*(\d+[.)]|[-*])\s*", "", l).strip() for l in text.split("\n")]
            raw.append([l for l in lines if l][:5])
        print(f"  {min(k + args.batch, len(pairs))}/{len(pairs)}", flush=True)
    del model
    torch.cuda.empty_cache()

    from sentence_transformers import SentenceTransformer
    emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
    reasons, kept = {}, 0
    results = dict(previous)
    if True:
        for p, cands in zip(pairs, raw):
            good = []
            if cands:
                E = emb.encode([p["question"]] + cands, normalize_embeddings=True)
                for c, s in zip(cands, (E[1:] @ E[0]).tolist()):
                    r = valid(c, p["question"], s, p["answer"])
                    if r is None and c not in good and len(good) < 3:
                        good.append(c)
                    else:
                        reasons[r or "duplicate"] = reasons.get(r or "duplicate", 0) + 1
            kept += len(good)
            results[p["id"]] = {"id": p["id"], "question": p["question"], "paraphrases": good}
    with open(ROOT / "data/qa/paraphrases.jsonl", "w", encoding="utf-8") as f:
        for r in results.values():
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n0 = sum(1 for c in raw if not c)
    print(f"{kept} rewordings kept for {len(pairs)} questions ({kept / len(pairs):.2f} per question); "
          f"rejected: {reasons}; empty outputs: {n0}")


assert anchors("For glycol dehydrators, what reduces emissions?", "Flash tanks on glycol dehydrators reduce it.") == {"glycol", "dehydrator"}
assert must_keep("What is the EF for Gas, Valve, GV in OGMP Table 2.4 at 4.9 scfh?") >= {"Table 2.4", "4.9", "EF", "GV"}

if __name__ == "__main__":
    main()
