# Base vs fine-tuned: Qwen2.5-1.5B-Instruct on OGMP QA (data/qa_v2/test_seen.jsonl, n = 70)

| Metric | Base | Fine-tuned | Change (95% CI) |
|---|---|---|---|
| BLEU (sacreBLEU corpus, 0-100) | 0.776 | 41.582 | +39.7 [+29.8, +50.2] (sentence-level) |
| ROUGE-L F1 | 0.097 | 0.531 | +0.433 [+0.347, +0.524] |
| Relevance (embedding cosine) | 0.714 | 0.822 | +0.108 [+0.074, +0.144] |
| Answer length (words) | 169.957 | 22.386 |  |
