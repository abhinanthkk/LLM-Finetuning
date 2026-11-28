# v2 knowledge-injection fine-tuning: results

Model: Qwen2.5-1.5B-Instruct, QLoRA r=16, trained on 2,638 examples (2,075 QA wordings of 694 facts + 563 document passages).
Training: 1,500 steps (4.5 epochs), early stopped; best validation loss 0.118 at step 1300 (base model 2.867). Peak GPU 2.84 GiB, 85 min.
Curves: `loss_curves.png`. Validation loss sits below training loss because training batches also contain the
document-passage examples (long verbatim text, harder to predict); validation is QA only.

## Test 1: trained facts, new wording (70 hand-written rewordings, never trained on)

| Metric | Base | Fine-tuned v2 | Change (95% CI) |
|---|---|---|---|
| BLEU (corpus) | 0.8 | 41.6 | +39.7 [+29.8, +50.2] |
| ROUGE-L F1 | 0.097 | 0.531 | +0.433 [+0.347, +0.524] |
| Relevance (embedding cosine) | 0.714 | 0.822 | +0.108 [+0.074, +0.144] |
| Answer length (words) | 170 | 22 | |
| **Manual grade: correct / partial / wrong** | **0 / 4 / 66** | **32 / 14 / 24** | |

v1 for comparison (test facts never trained on): fine-tuned 6/16/48, base 3/22/45.

Fine-tuned by question type (C/P/W): lookup 13/3/8, why 7/2/9, definition 5/3/0, quantification 2/4/1, comparison 4/2/4, mitigation 1/0/2.

## Test 2: unseen control (20 facts in no training QA pair; only in passage text)

| Metric | Base | Fine-tuned v2 |
|---|---|---|
| ROUGE-L F1 | 0.083 | 0.261 |
| Relevance | 0.726 | 0.736 (CI includes 0) |
| Manual grade C / P / W | 1 / 3 / 16 | 1 / 4 / 15 |

Facts the model only read as passages are not recalled as answers: reading the text is not enough for a 1.5B model.

## Error analysis (fine-tuned, test 1)

1. **Table numbers mixed up between rows** (Table 2.4 connector and open-ended line, Table 4.2 Storage, Table 9.2): the right
   format and units are kept, but the value comes from a neighbouring row. Numbers in prose are also mixed up (pump savings, smart well cost, REC reduction).
2. **Fluent but wrong answers in the right style** (Mitigation Option B, stabilization tower, REC piping): the model answers confidently in the trained style, so these are hard to spot.
3. **Partly correct answers with an invented extra clause** (chemical injection pumps "flame arrestors", turbine meter flow range).
4. **Missing qualifier** (Table 2.3 row label dropped, but numbers correct).
The base model's errors are different: long generic answers (170 words, often cut off at the 256-token limit), invented document names, wrong acronym expansions ("Virtual Recirculation Unit").

Grades are in `manual_grades_seen.csv` and `manual_grades_unseen.csv`. One grader (Claude). Base answers were graded on their content against the reference; they contain almost no OGMP-specific facts.
