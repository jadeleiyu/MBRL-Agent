from transformers import AutoModelForCausalLM
from peft import PeftModel
import os

# BASE = "openai/gpt-oss-20b"
# ADAPTER_DIR = "/checkpoint/multimodal-reasoning/jadeleiyu/mbrl_agent/wm_sft/gpt-oss-20b_webarena_-1_sft_lora_16"
# REPO_ID = "jadeleiyu/gpt-oss-20b_WM_webarena_sft_lora_16"

# ADAPTER_DIR = "/checkpoint/multimodal-reasoning/jadeleiyu/mbrl_agent/agent_sft/gpt-oss-20b_nnetnav-wa_-1_sft_lora_16"
# REPO_ID = "jadeleiyu/gpt-oss-20b_agent_nnetnav-wa_sft_lora_16"

BASE = "unsloth/gpt-oss-120b-BF16"
ADAPTER_DIR = "/checkpoint/multimodal-reasoning/jadeleiyu/mbrl_agent/wm_sft/"
ENVS = ['live', 'wa']

for env in ENVS:
    adapter_path = os.path.join(ADAPTER_DIR, f"gpt-oss-120b-{env}-lf")
    repo_id = f"jadeleiyu/gpt-oss-120b_WM_{env}"

    base = AutoModelForCausalLM.from_pretrained(BASE, trust_remote_code=True)
    peft_model = PeftModel.from_pretrained(base, adapter_path)

    # Option A: push whatever is in ADAPTER_DIR
    peft_model.push_to_hub(repo_id, commit_message="Add LoRA adapter")

# (If you need to save locally first)
# peft_model.save_pretrained(ADAPTER_DIR)
# peft_model.push_to_hub(REPO_ID)


# from datasets import load_dataset

# ds = load_dataset("json", data_files={"train": "/home/jadeleiyu/projects/mbrl_agent/world_model/sft_data/wm_wa_sft.json"})
# ds.push_to_hub("jadeleiyu/WM-webarena-sft")
