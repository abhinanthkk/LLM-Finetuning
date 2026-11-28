"""Stage 4: base vs fine-tuned comparison on data/qa_v2/test_seen.jsonl (or --test data/qa_v2/test_unseen.jsonl).

Both models get the same system prompt, chat template and greedy decoding; the only difference is the LoRA
adapter (the base model is the same weights with the adapter switched off).

Metrics per answer, against the reference answer:
  BLEU      sacreBLEU (corpus score, 0-100; sentence scores for the CSV)
  ROUGE-L   F1 of the longest common word subsequence (rouge-score, with stemming)
  relevance cosine similarity of sentence embeddings (BAAI/bge-small-en-v1.5)
  judge     optional (--judge MODEL): a blind 1-5 correctness score from an LLM, via the make_qa endpoint
Differences come with a 95% paired-bootstrap confidence interval (the test set is small).

Outputs (outputs_v2/): eval_answers.csv (one row per question, for error analysis), eval_results.json, eval_report.md
usage: python scripts/eval.py [--judge gemini-3.5-flash] [--limit N]
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs_v2"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
EMBED = "BAAI/bge-small-en-v1.5"


@torch.no_grad()
def generate(model, tok, prompts, batch=8, max_new=256):
    tok.padding_side = "left"                                  # decoder-only models generate from the right edge
    out = []
    for k in range(0, len(prompts), batch):
        enc = tok.apply_chat_template(prompts[k:k + batch], add_generation_prompt=True, return_tensors="pt",
                                      padding=True, return_dict=True).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                             pad_token_id=tok.pad_token_id)
        out += [tok.decode(g[enc["input_ids"].shape[1]:], skip_special_tokens=True).strip() for g in gen]
        print(f"  {min(k + batch, len(prompts))}/{len(prompts)}", flush=True)
    return out


def bootstrap_diff(a, b, n=2000, seed=0):
    """95% CI of mean(b - a) over questions, resampling questions with replacement."""
    rng = np.random.default_rng(seed)
    d = np.asarray(b) - np.asarray(a)
    means = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return float(d.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


JUDGE_PROMPT = """You are grading answers from a question-answering assistant about methane emissions in oil and gas
(OGMP technical guidance). For each item, compare the ANSWER with the REFERENCE answer and score correctness:
5 = fully correct and complete; 4 = correct with a minor omission; 3 = partly correct;
2 = mostly wrong, vague or off-topic; 1 = wrong or contains invented facts or numbers.
Judge facts, not wording or length. Return only a JSON object mapping each item id to its score, e.g. {{"7": 4}}.

{items}"""


def judge(items, model, batch=40):
    """items: list of (key, question, reference, answer). Returns {key: score}. Shuffled so the grader cannot tell
    which model wrote an answer."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_qa
    import httpx
    import os
    key = os.environ.get("QA_API_KEY") or os.environ.get("GEMINI_API_KEY")
    client = httpx.Client(base_url=make_qa.BASE_URL, timeout=300, headers={"Authorization": f"Bearer {key}"})
    items = items[:]
    random.Random(1).shuffle(items)
    scores = {}
    for k in range(0, len(items), batch):
        group = items[k:k + batch]
        text = "\n\n".join(f"ITEM {i}\nQUESTION: {q}\nREFERENCE: {r}\nANSWER: {a}" for i, (_, q, r, a) in enumerate(group))
        raw = make_qa.call_llm(client, model, JUDGE_PROMPT.format(items=text))
        raw = raw[raw.find("{"):raw.rfind("}") + 1]
        got = json.loads(raw)
        for i, (key_, *_rest) in enumerate(group):
            s = got.get(str(i))
            if isinstance(s, (int, float)) and 1 <= s <= 5:
                scores[key_] = float(s)
        print(f"  judged {min(k + batch, len(items))}/{len(items)}", flush=True)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", help="LLM model id for blind 1-5 grading (optional)")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--adapter", default="outputs_v2/adapter")
    ap.add_argument("--test", default="data/qa_v2/test_seen.jsonl")
    ap.add_argument("--out", default="outputs_v2")
    ap.add_argument("--tag", default="", help="suffix for output file names, e.g. _seen")
    args = ap.parse_args()

    global OUT
    OUT = ROOT / args.out
    OUT.mkdir(exist_ok=True)
    test = [json.loads(l) for l in open(ROOT / args.test, encoding="utf-8")][:args.limit]
    prompts = [r["prompt"] for r in test]
    refs = [r["completion"][0]["content"] for r in test]

    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, device_map={"": 0}, dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    model = PeftModel.from_pretrained(base, str(ROOT / args.adapter)).eval()
    print("generating: fine-tuned")
    tuned = generate(model, tok, prompts)
    print("generating: base (adapter disabled)")
    with model.disable_adapter():
        basel = generate(model, tok, prompts)
    del model, base
    torch.cuda.empty_cache()

    import sacrebleu
    from rouge_score import rouge_scorer
    from sentence_transformers import SentenceTransformer
    rs = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    emb = SentenceTransformer(EMBED, device="cuda")
    E_ref = emb.encode(refs, normalize_embeddings=True)

    per, results = {}, {}
    for name, ans in (("base", basel), ("finetuned", tuned)):
        E = emb.encode(ans, normalize_embeddings=True)
        per[name] = {
            "bleu": [sacrebleu.sentence_bleu(a, [r]).score for a, r in zip(ans, refs)],
            "rougeL": [rs.score(r, a)["rougeL"].fmeasure for a, r in zip(ans, refs)],
            "relevance": [float(x) for x in (E * E_ref).sum(1)],
            "words": [len(a.split()) for a in ans],
        }
        results[name] = {"BLEU (corpus)": sacrebleu.corpus_bleu(ans, [refs]).score,
                         **{k: float(np.mean(v)) for k, v in per[name].items() if k != "bleu"}}

    if args.judge:
        print(f"judging with {args.judge}")
        items = [(f"{n}:{i}", test[i]["prompt"][-1]["content"], refs[i], ans[i])
                 for n, ans in (("base", basel), ("finetuned", tuned)) for i in range(len(test))]
        sc = judge(items, args.judge)
        for n in ("base", "finetuned"):
            per[n]["judge"] = [sc.get(f"{n}:{i}", float("nan")) for i in range(len(test))]
            results[n]["judge (1-5)"] = float(np.nanmean(per[n]["judge"]))

    diffs = {m: bootstrap_diff(per["base"][m], per["finetuned"][m]) for m in per["base"]
             if m != "words" and not np.isnan(per["base"][m]).any() and not np.isnan(per["finetuned"][m]).any()}
    results["difference (finetuned - base), mean and 95% CI"] = diffs
    results["n_test"] = len(test)
    json.dump(results, open(OUT / f"eval_results{args.tag}.json", "w"), indent=2)

    with open(OUT / f"eval_answers{args.tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        cols = [m for m in per["base"]]
        w.writerow(["id", "doc", "type", "source", "question", "reference", "base_answer", "finetuned_answer"]
                   + [f"base_{m}" for m in cols] + [f"ft_{m}" for m in cols])
        for i, r in enumerate(test):
            w.writerow([r["id"], r["doc"], r["type"], r["source"], r["prompt"][-1]["content"], refs[i], basel[i], tuned[i]]
                       + [round(per["base"][m][i], 4) for m in cols] + [round(per["finetuned"][m][i], 4) for m in cols])

    rows = [("BLEU (sacreBLEU corpus, 0-100)", "BLEU (corpus)", "bleu"), ("ROUGE-L F1", "rougeL", "rougeL"),
            ("Relevance (embedding cosine)", "relevance", "relevance"), ("LLM judge (1-5)", "judge (1-5)", "judge"),
            ("Answer length (words)", "words", None)]
    md = [f"# Base vs fine-tuned: Qwen2.5-1.5B-Instruct on OGMP QA ({args.test}, n = {len(test)})\n",
          "| Metric | Base | Fine-tuned | Change (95% CI) |", "|---|---|---|---|"]
    for label, key, dkey in rows:
        if key not in results["base"]:
            continue
        ci = ""
        if dkey in diffs:
            m, lo, hi = diffs[dkey]
            ci = f"{m:+.3f} [{lo:+.3f}, {hi:+.3f}]" if dkey != "bleu" else f"{m:+.1f} [{lo:+.1f}, {hi:+.1f}] (sentence-level)"
        md.append(f"| {label} | {results['base'][key]:.3f} | {results['finetuned'][key]:.3f} | {ci} |")
    (OUT / f"eval_report{args.tag}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
