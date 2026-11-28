# OGMP methane QA: fine-tuning a small LLM on oil and gas guidance

A domain question-answering assistant for oil and gas engineers, made by fine-tuning **Qwen2.5-1.5B-Instruct** with
**QLoRA** on the OGMP (Oil & Gas Methane Partnership) technical guidance documents. Fine-tuning only, no retrieval:
the model answers from what it learned.

## Results

Test: 70 facts from training, each asked with a new hand-written wording the model never saw.

| | Base | Fine-tuned |
|---|---|---|
| BLEU | 0.8 | 41.6 |
| ROUGE-L F1 | 0.097 | 0.531 |
| Relevance (embedding cosine) | 0.714 | 0.822 |
| Manual grade: correct / partial / wrong | 0 / 4 / 66 | 32 / 14 / 24 |

On 20 facts held out of all training QA pairs, both models get 1 of 20 right: the model learns facts it is trained
to answer, not facts it only read as passages. Full results, training curves and error analysis:
[`outputs_v2/RESULTS.md`](outputs_v2/RESULTS.md).

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
```

Download the 10 OGMP technical guidance PDFs (TGD1 to TGD9 and Appendix A) from the CCAC OGMP page into `OGMP docs/`.
QA generation with Gemini needs `GEMINI_API_KEY=...` in a `.env` file (git-ignored); the generated pairs are
already in `data/qa/`, so this step is optional.

## Pipeline

| Stage | Script | Output |
|---|---|---|
| 0 | `scripts/smoke_test.py` | memory and speed check on a 4 GB GPU |
| 1 | `scripts/extract.py` | `data/chunks.jsonl`, `data/extracted/*.md`: PDF text and tables as clean sections |
| 2 | `scripts/make_qa.py --from-cache` | `data/qa/qa_raw.jsonl`: 714 evidence-checked QA pairs (table templates, Gemini, written in chat via `claude_batch.py`) |
| 2 | `scripts/check_qa.py` | audit of every pair |
| 2 | `scripts/paraphrase.py` | `data/qa/paraphrases.jsonl`: question rewordings (Qwen3-1.7B) |
| 2 | `scripts/select_v2.py`, `scripts/build_v2.py` | `data/qa_v2/`: train, validation, seen-fact test, unseen-fact test |
| 3 | `scripts/train.py` | `outputs_v2/adapter`, `train_log.jsonl`, `loss_curves.png` |
| 4 | `scripts/eval.py` (and `--test data/qa_v2/test_unseen.jsonl --tag _unseen`) | metrics, answers CSV, report |
| 5 | `scripts/app.py` | Gradio app at http://127.0.0.1:7860 |

`split_qa.py` holds the shared system prompt and fact grouping used by stages 2 and later.

## The app

```bash
cd scripts && ../.venv/bin/python app.py
```

Ask a question: the base and fine-tuned answers appear side by side. If the question matches one in the checked
dataset (embedding similarity of 0.85 or more), that pair's answer from the PDF is the reference for BLEU, ROUGE-L and
relevance; otherwise only a grounding score is shown.

## Training setup

4-bit NF4 base with double quantisation, bf16 compute, LoRA r=16 / alpha=32 on all attention and MLP projections
(18.5M trainable parameters, 1.2%), loss on answer tokens only, AdamW 2e-4 with cosine schedule, early stopping on
validation loss. Peak GPU memory 2.84 GiB, 85 minutes on an RTX 2050.
