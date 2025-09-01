from transformers import AutoModelForCausalLM
from peft import PeftModel

BASE = "openai/gpt-oss-20b"
ADAPTER_DIR = "/checkpoint/multimodal-reasoning/jadeleiyu/mbrl_agent/wm_sft/gpt-oss-20b_webarena_sft_lora_16/"
REPO_ID = "jadeleiyu/gpt-oss-20b_WM_webarena_sft_lora_16"

base = AutoModelForCausalLM.from_pretrained(BASE, trust_remote_code=True)
peft_model = PeftModel.from_pretrained(base, ADAPTER_DIR)

# Option A: push whatever is in ADAPTER_DIR
peft_model.push_to_hub(REPO_ID, commit_message="Add LoRA adapter")

# (If you need to save locally first)
# peft_model.save_pretrained(ADAPTER_DIR)
# peft_model.push_to_hub(REPO_ID)


