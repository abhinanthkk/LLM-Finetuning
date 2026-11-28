"""Stage 5: Gradio QA assistant. Base and fine-tuned answers side by side, with per-query evaluation metrics.

The models answer on their own (no retrieval). Only for scoring, the question is matched to the closest question in the
checked QA dataset (originals + rewordings); if similarity >= MATCH, that pair's answer is the reference for BLEU,
ROUGE-L and relevance. Otherwise there is no reference, and only a grounding score is shown: the answer's similarity
to the closest OGMP document chunk (how much it sounds like the guidance, not whether it is correct).
usage: python scripts/app.py [--adapter outputs_v2/adapter]   then open http://127.0.0.1:7860
"""
import argparse
import html
import json
from pathlib import Path

import gradio as gr
import sacrebleu
import torch
from peft import PeftModel
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from split_qa import SYSTEM

ROOT = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
MATCH = 0.85   # calibrated: hand rewordings of dataset questions score 0.83-0.97, off-topic questions <= 0.77

ap = argparse.ArgumentParser()
ap.add_argument("--adapter", default="outputs_v2/adapter")
ap.add_argument("--share", action="store_true", help="also create a public gradio.live link")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(MODEL)
tok.padding_side = "left"
base = AutoModelForCausalLM.from_pretrained(MODEL, device_map={"": 0}, dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
model = PeftModel.from_pretrained(base, str(ROOT / args.adapter)).eval()

emb = SentenceTransformer("BAAI/bge-small-en-v1.5", device="cuda")
rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
pairs = [json.loads(l) for l in open(ROOT / "data/qa/qa_raw.jsonl", encoding="utf-8")]
para = {r["id"]: r["paraphrases"] for r in map(json.loads, open(ROOT / "data/qa/paraphrases.jsonl", encoding="utf-8"))}
known_q, known_pair = [], []
for p in pairs:
    for q in [p["question"]] + para.get(p["id"], []):
        known_q.append(q)
        known_pair.append(p)
Q_EMB = emb.encode(known_q, normalize_embeddings=True, convert_to_tensor=True)
chunks = [json.loads(l) for l in open(ROOT / "data/chunks.jsonl", encoding="utf-8")]
C_EMB = emb.encode([c["text"] for c in chunks], normalize_embeddings=True, convert_to_tensor=True)


@torch.no_grad()
def generate(question, adapter_on):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
    run = lambda: model.generate(**enc, max_new_tokens=256, do_sample=False, temperature=None, top_p=None, top_k=None,
                                 pad_token_id=tok.pad_token_id)
    if adapter_on:
        out = run()
    else:
        with model.disable_adapter():
            out = run()
    new = out[0][enc["input_ids"].shape[1]:]
    text = tok.decode(new, skip_special_tokens=True).strip()
    return text + ("\u2009\u2026" if tok.eos_token_id not in new.tolist() and len(new) >= 256 else "")


def metrics(answers, ref):
    """Rows of (name, hint, base value, fine-tuned value, decimals, higher_is_better); ref=None drops the reference metrics."""
    A = emb.encode(answers, normalize_embeddings=True, convert_to_tensor=True)
    ground = (A @ C_EMB.T).max(1).values.tolist()
    rows = []
    if ref is not None:
        R = emb.encode([ref], normalize_embeddings=True, convert_to_tensor=True)
        rel = (A @ R.T).squeeze(1).tolist()
        bleu = [sacrebleu.sentence_bleu(a, [ref]).score for a in answers]
        rl = [rouge.score(ref, a)["rougeL"].fmeasure for a in answers]
        rows += [("BLEU", "n-gram overlap with the reference, 0 to 100", *bleu, 1, True),
                 ("ROUGE-L F1", "longest shared word sequence, 0 to 1", *rl, 3, True),
                 ("Relevance", "meaning similarity to the reference, 0 to 1", *rel, 3, True)]
    rows += [("Grounding", "similarity to the closest OGMP passage, 0 to 1", *ground, 3, True),
             ("Answer length", "words", len(answers[0].split()), len(answers[1].split()), 0, None)]
    return rows


E = html.escape
SHELL = ("<div class='bar'><span class='wordmark'>OGMP methane QA</span>"
         "<span class='model'>Qwen2.5-1.5B-Instruct, base and QLoRA fine-tuned</span></div>")
EMPTY = ("<div class='state'><h2>No question yet</h2><p>Ask about methane emissions in oil and gas operations, or "
         "pick an example. Both models answer on their own, and their scores appear here.</p></div>")
SKELETON = "<div class='skeleton' aria-hidden='true'><span></span><span></span><span></span><span></span></div>"


def pane(title, who, answer, cls=""):
    if answer is None:
        body = f"<p class='waiting' role='status'>Writing the answer</p>{SKELETON}"
    else:
        n = len(answer.split())
        cut = ", <span class='cut-off'>cut off at the 256-token limit</span>" if answer.endswith("\u2026") else ""
        body = f"<div class='answer'>{E(answer)}</div><p class='count'>{n} word{'s' * (n != 1)}{cut}</p>"
    return f"<article class='pane {cls}'><h2>{title}</h2><p class='who'>{who}</p>{body}</article>"


def page(basel=None, tuned=None, score=""):
    return ("<div class='panes'>"
            + pane("Base model", "Qwen2.5-1.5B-Instruct, adapter off", basel)
            + pane("Fine-tuned model", "Same weights with the OGMP LoRA adapter", tuned, "pane--tuned")
            + "</div>" + score)


def scorecard(question, basel, tuned):
    sims = (emb.encode([question], normalize_embeddings=True, convert_to_tensor=True) @ Q_EMB.T)[0]
    best = int(sims.argmax())
    sim = float(sims[best])
    if sim >= MATCH:
        p = known_pair[best]
        ref = (f"<div class='reference'><p class='matched'>Matched dataset question, similarity "
               f"<span class='sim'>{sim:.2f}</span>: {E(known_q[best])}</p>"
               f"<p class='ref'>{E(p['answer'])}</p>"
               f"<p class='src'>Reference answer from {E(p['doc'])}, section {E(p['section'])}</p></div>")
        rows, note = metrics([basel, tuned], p["answer"]), ""
    else:
        ref = (f"<div class='reference reference--none'><p class='ref'>No reference answer in the dataset for this "
               f"question. The closest one, at similarity <span class='sim'>{sim:.2f}</span> (a match needs {MATCH}), is: "
               f"{E(known_q[best])}</p></div>")
        rows = metrics([basel, tuned], None)
        note = ("<p class='note'>BLEU, ROUGE-L and relevance need a reference answer, so only grounding is shown. "
                "Grounding says how close an answer is to the OGMP text, not whether it is correct.</p>")
    body = ""
    for name, hint, b, t, dec, higher in rows:
        better_b = higher is not None and round(b, dec) > round(t, dec)
        better_t = higher is not None and round(t, dec) > round(b, dec)
        cell = lambda v, win: f"<td class='{'better' if win else ''}'>{v:.{dec}f}</td>"
        body += (f"<tr><th scope='row'>{name}<span class='hint'>{hint}</span></th>"
                 f"{cell(b, better_b)}{cell(t, better_t)}</tr>")
    table = ("<table class='metrics'><caption class='visually-hidden'>Evaluation metrics for this question</caption>"
             "<thead><tr><th scope='col'>Metric</th><th scope='col'>Base</th><th scope='col'>Fine-tuned</th></tr></thead>"
             f"<tbody>{body}</tbody></table>")
    return f"<section class='score'><h2>Scorecard</h2>{ref}{table}{note}</section>"


def ask(question):
    question = (question or "").strip()
    if not question:
        yield EMPTY
        return
    try:
        yield page()
        tuned = generate(question, True)
        yield page(None, tuned)
        basel = generate(question, False)
        yield page(basel, tuned, scorecard(question, basel, tuned))
    except Exception as e:  # shown in the page, so a failed run never looks like an empty answer
        torch.cuda.empty_cache()
        yield (f"<div class='state state--error' role='alert'><h2>The answer could not be generated</h2>"
               f"<p>{E(type(e).__name__)}: {E(str(e)[:300])}</p><p>Close other programs using the GPU and ask again.</p></div>")


EXAMPLES = ["How much of the methane and VOCs carried by the TEG does a flash tank separator recover?",
            "What kinds of compressors can handle wet casinghead gas?",
            "Why is sending casinghead gas to a flare considered the last option?",
            "What is a heat exchanger?"]
UI = Path(__file__).resolve().parent / "ui"
CSS = "\n".join((UI / f).read_text() for f in ("tokens.css", "base.css", "app.css"))
CSS = "\n".join(l for l in CSS.splitlines() if not l.startswith("@import"))
FONTS = ('<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Familjen+Grotesk:wght@500;600'
         '&family=IBM+Plex+Sans:wght@400;500&family=JetBrains+Mono:wght@400&display=swap">')

with gr.Blocks(title="OGMP methane QA: base vs fine-tuned") as demo:
    gr.HTML(SHELL)
    gr.HTML("<div class='ask-head'><h1>Ask the OGMP guidance</h1><p>Both models answer without seeing any document. "
            "When the question matches one in the checked dataset, its answer from the PDF is used to score them.</p></div>")
    with gr.Row(elem_classes=["ask-row"]):
        q = gr.Textbox(label="Question", placeholder="What share of methane does a glycol flash tank capture?",
                       elem_id="question", lines=1, max_lines=4, scale=5, autofocus=True)
        btn = gr.Button("Ask", elem_classes=["key", "key--primary"], scale=1, min_width=120)
    gr.HTML("<p class='tray-label'>Examples</p>")
    with gr.Row(elem_classes=["tray"]):
        ex = [gr.Button(t, elem_classes=["key"], size="sm") for t in EXAMPLES]
    out = gr.HTML(EMPTY, elem_classes=["result"])
    btn.click(ask, q, out, show_progress="hidden")
    q.submit(ask, q, out, show_progress="hidden")
    for b, t in zip(ex, EXAMPLES):
        b.click(lambda t=t: t, None, q).then(ask, q, out, show_progress="hidden")

if __name__ == "__main__":
    demo.queue().launch(share=args.share, css=CSS, head=FONTS, footer_links=[])
