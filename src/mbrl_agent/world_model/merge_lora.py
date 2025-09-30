from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch, os


BASE = "unsloth/gpt-oss-120b-BF16"   
MODEL_DIR = "/checkpoint/multimodal-reasoning/jadeleiyu/mbrl_agent/wm_sft/"

adapter_path = os.path.join(MODEL_DIR, f"gpt-oss-120b-llamafactory/checkpoint-500")
merged_path = os.path.join(MODEL_DIR, f"gpt-oss-120b-llamafactory-merged-500")
os.makedirs(merged_path, exist_ok=True)

# 1) Load base in full precision (no 4-bit) so weights can be merged
model = AutoModelForCausalLM.from_pretrained(
    BASE,
    torch_dtype=torch.bfloat16,          # or torch.float16 / float32 (CPU)
    device_map="auto",                   # spread across GPUs / CPU offload if needed
    trust_remote_code=True,
)
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)

# 2) Attach LoRA and merge into the base weights
model = PeftModel.from_pretrained(model, adapter_path)
model = model.merge_and_unload()         # removes LoRA layers; weights now baked in

# 3) Save a normal Transformers checkpoint
model.save_pretrained(merged_path, safe_serialization=True)  # writes model.safetensors
tok.save_pretrained(merged_path)

print(f"Merged model saved to: {merged_path}")



