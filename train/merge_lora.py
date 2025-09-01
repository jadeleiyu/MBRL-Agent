from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, os

BASE = "openai/gpt-oss-20b"                     # HF id or local path of your base
ADAPTER_DIR = "/checkpoint/multimodal-reasoning/jadeleiyu/mbrl_agent/gpt-oss-20b_nnetnav-wa_10000_sft_lora_16" # folder with adapter_config.json + adapter_model.safetensors
OUT = "/checkpoint/multimodal-reasoning/jadeleiyu/mbrl_agent/gpt-oss-20b_nnetnav-wa_10000_sft_lora_16_merged"

os.makedirs(OUT, exist_ok=True)

# 1) Load base in full precision (no 4-bit) so weights can be merged
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.bfloat16,          # or torch.float16 / float32 (CPU)
    device_map="auto",                   # spread across GPUs / CPU offload if needed
    trust_remote_code=True,
)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

# 2) Attach LoRA and merge into the base weights
model = PeftModel.from_pretrained(model, ADAPTER_DIR)
model = model.merge_and_unload()         # removes LoRA layers; weights now baked in

# 3) Save a normal Transformers checkpoint
model.save_pretrained(OUT, safe_serialization=True)  # writes model.safetensors
tok.save_pretrained(OUT)

print(f"Merged model saved to: {OUT}")
