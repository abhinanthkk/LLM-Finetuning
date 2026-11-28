# Base vs fine-tuned: Qwen2.5-1.5B-Instruct on OGMP QA (data/qa_v2/test_unseen.jsonl, n = 20)

| Metric | Base | Fine-tuned | Change (95% CI) |
|---|---|---|---|
| BLEU (sacreBLEU corpus, 0-100) | 0.870 | 17.677 | +11.3 [+2.6, +24.1] (sentence-level) |
| ROUGE-L F1 | 0.083 | 0.261 | +0.178 [+0.099, +0.286] |
| Relevance (embedding cosine) | 0.726 | 0.736 | +0.010 [-0.025, +0.050] |
| Answer length (words) | 174.800 | 25.950 |  |
