"""Stage 0: does QLoRA on Qwen2.5-1.5B fit the local GPU? 10 steps, batch 1.
usage: smoke_test.py SEQ_LEN ANSWER_TOKENS   (loss only on the last ANSWER_TOKENS, like completion-only SFT)"""
import sys, time, torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

seq, ans = int(sys.argv[1]), int(sys.argv[2])
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct", device_map={"": 0}, dtype=torch.bfloat16,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16))
model.config.use_cache = False  # a KV cache silently disables gradient checkpointing
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads()
model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
model.train()
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
ids = torch.randint(100, 20000, (1, seq), device="cuda")
torch.cuda.reset_peak_memory_stats(); t0 = time.time()
for step in range(10):
    # vocab is 151,936: project only the answer positions to logits, never the whole sequence
    h = model.base_model.model.model(input_ids=ids, use_cache=False).last_hidden_state[:, -ans - 1:-1]
    logits = model.base_model.model.lm_head(h).float()
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), ids[:, -ans:].reshape(-1))
    loss.backward(); opt.step(); opt.zero_grad()
torch.cuda.synchronize()
print(f"seq {seq}, answer {ans}: loss {loss.item():.2f}, peak {torch.cuda.max_memory_allocated()/2**30:.2f} GiB, "
      f"{(time.time()-t0)/10:.2f} s/step")
