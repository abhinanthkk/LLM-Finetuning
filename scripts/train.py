"""Stage 3: QLoRA fine-tuning of Qwen2.5-1.5B-Instruct on data/qa_v2/train.jsonl, validated on data/qa_v2/val.jsonl.

Outputs (in outputs_v2/):
  adapter/          best LoRA adapter (lowest validation loss)
  train_log.jsonl   one line per logged step: step, epoch, train_loss, val_loss, lr
  loss_curves.png   training and validation loss curves
usage:
  python scripts/train.py --max-steps 10     quick check that it runs and fits in memory
  python scripts/train.py                    full run

Memory notes from the stage 0 test on a 4 GB RTX 2050 (see the study guide):
  - no prepare_model_for_kbit_training: it casts the 151k-token embedding to fp32 (~0.9 GB)
  - use_cache=False: a KV cache silently disables gradient checkpointing
  - logits are computed only at answer positions, never for the whole sequence x 151,936-token vocabulary
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
OUT = ROOT / "outputs_v2"


def load(split, tok, max_len, data_dir=ROOT / "data/qa_v2"):
    """Token ids plus labels: -100 on the prompt (system + question), the answer tokens as targets."""
    rows = []
    for line in open(Path(data_dir) / f"{split}.jsonl", encoding="utf-8"):
        r = json.loads(line)
        p = tok.apply_chat_template(r["prompt"], add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"]
        f = tok.apply_chat_template(r["prompt"] + r["completion"], tokenize=True, return_dict=True)["input_ids"]
        assert f[:len(p)] == p, f"prompt is not a prefix of the full example: {r['id']}"
        f = f[:max_len]
        rows.append({"ids": f, "labels": [-100] * len(p) + f[len(p):]})
    return rows


def batches(rows, size, pad_id, shuffle, rng):
    order = list(range(len(rows)))
    if shuffle:
        rng.shuffle(order)
    for k in range(0, len(order), size):
        group = [rows[i] for i in order[k:k + size]]
        n = max(len(r["ids"]) for r in group)
        ids = torch.tensor([r["ids"] + [pad_id] * (n - len(r["ids"])) for r in group])
        labels = torch.tensor([r["labels"] + [-100] * (n - len(r["labels"])) for r in group])
        mask = torch.tensor([[1] * len(r["ids"]) + [0] * (n - len(r["ids"])) for r in group])
        yield ids, labels, mask


def answer_loss(model, ids, labels, mask):
    """Mean cross-entropy over answer tokens. Hidden state t predicts token t+1, so shift by one, then project
    only the answer positions through the 151k-token lm_head."""
    inner = model.base_model.model
    h = inner.model(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
    target = labels[:, 1:]
    keep = target != -100
    logits = inner.lm_head(h[:, :-1][keep]).float()
    return F.cross_entropy(logits, target[keep])


@torch.no_grad()
def evaluate(model, rows, size, pad_id):
    model.eval()
    total, count = 0.0, 0
    for ids, labels, mask in batches(rows, size, pad_id, False, None):
        n = int((labels[:, 1:] != -100).sum())
        total += answer_loss(model, ids.cuda(), labels.cuda(), mask.cuda()).item() * n
        count += n
    model.train()
    return total / count


def plot(log, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    tr = [(e["step"], e["train_loss"]) for e in log if "train_loss" in e]
    va = [(e["step"], e["val_loss"]) for e in log if "val_loss" in e]
    plt.figure(figsize=(7, 4))
    plt.plot(*zip(*tr), color="black", alpha=0.25, linewidth=0.8, label="training loss (per batch)")
    k = 10
    if len(tr) > k:
        sm = [(tr[i][0], sum(v for _, v in tr[i - k + 1:i + 1]) / k) for i in range(k - 1, len(tr))]
        plt.plot(*zip(*sm), color="black", label=f"training loss ({k}-step average)")
    if va:
        plt.plot(*zip(*va), "--o", label="validation loss", color="black", markersize=3)
    plt.xlabel("optimizer step")
    plt.ylabel("loss (answer tokens)")
    plt.title("QLoRA fine-tuning: Qwen2.5-1.5B-Instruct on OGMP QA")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--max-len", type=int, default=256)        # longest example is 228 tokens
    ap.add_argument("--eval-every", type=int, default=50)
    ap.add_argument("--patience", type=int, default=4, help="evaluations without improvement before stopping")
    ap.add_argument("--max-steps", type=int, help="stop after N steps (quick check)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", default="data/qa_v2", help="folder with train.jsonl and val.jsonl")
    ap.add_argument("--out", default="outputs_v2", help="folder for adapter, log and curves")
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    global OUT
    OUT = ROOT / args.out
    OUT.mkdir(exist_ok=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    train, val = load("train", tok, args.max_len, ROOT / args.data), load("val", tok, args.max_len, ROOT / args.data)
    print(f"train {len(train)} examples, val {len(val)}")

    model = AutoModelForCausalLM.from_pretrained(MODEL, device_map={"": 0}, dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.print_trainable_parameters()
    model.train()

    steps_per_epoch = math.ceil(len(train) / args.batch)
    total = args.max_steps or steps_per_epoch * args.epochs
    warmup = max(1, int(0.03 * total))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warmup if s < warmup
                                              else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup))))

    log, best, bad, step, t0 = [], float("inf"), 0, 0, time.time()
    first_val = evaluate(model, val, args.batch, tok.pad_token_id)
    log.append({"step": 0, "epoch": 0, "val_loss": first_val})
    print(f"step 0: val loss {first_val:.3f} (base model, before training)")
    torch.cuda.reset_peak_memory_stats()
    stop = False
    for epoch in range(1, args.epochs + 1):
        for ids, labels, mask in batches(train, args.batch, tok.pad_token_id, True, rng):
            loss = answer_loss(model, ids.cuda(), labels.cuda(), mask.cuda())
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad()
            step += 1
            entry = {"step": step, "epoch": round(step / steps_per_epoch, 3), "train_loss": loss.item(),
                     "lr": sched.get_last_lr()[0]}
            if step % args.eval_every == 0 or step == total:
                entry["val_loss"] = evaluate(model, val, args.batch, tok.pad_token_id)
                if entry["val_loss"] < best - 1e-4:
                    best, bad = entry["val_loss"], 0
                    model.save_pretrained(OUT / "adapter")
                else:
                    bad += 1
                print(f"step {step}/{total} epoch {entry['epoch']:.2f}: train {loss.item():.3f} val {entry['val_loss']:.3f}"
                      f" (best {best:.3f}) | {time.time() - t0:.0f}s, peak GPU {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB",
                      flush=True)
            log.append(entry)
            if bad >= args.patience:
                print(f"early stop: validation loss has not improved for {args.patience} evaluations")
                stop = True
            if stop or step >= total:
                break
        if stop or step >= total:
            break

    with open(OUT / "train_log.jsonl", "w") as f:
        for e in log:
            f.write(json.dumps(e) + "\n")
    plot(log, OUT / "loss_curves.png")
    print(f"done: {step} steps in {time.time() - t0:.0f}s; best val loss {best:.3f} (base {first_val:.3f}); "
          f"adapter in {(OUT / 'adapter').relative_to(ROOT)}, curves in {(OUT / 'loss_curves.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
