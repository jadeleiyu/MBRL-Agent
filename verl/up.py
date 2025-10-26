from huggingface_hub import HfApi, HfFolder, upload_folder
import os

# === 基本配置 ===
HF_TOKEN = HfFolder.get_token()  # 自动读取登录token
repo_id = "DearSloth/Llama3.1-8b-rllm0.1-WA"  # 目标仓库
local_ckpt_path = "checkpoints/rllm-agent/7b-webarena_world_model/global_step_75/actor/checkpoint"  # 本地目录路径
commit_message = "Upload model checkpoint folder"

# === 初始化API ===
api = HfApi()

# 检查仓库是否存在，否则创建
try:
    api.repo_info(repo_id)
    print(f"✅ 仓库已存在：{repo_id}")
except Exception:
    api.create_repo(repo_id, private=True)
    print(f"🆕 创建新仓库：{repo_id}")

# === 上传整个文件夹 ===
print(f"🚀 开始上传目录：{local_ckpt_path}")
upload_folder(
    repo_id=repo_id,
    folder_path=local_ckpt_path,
    commit_message=commit_message,
    repo_type="model"
)

print(f"✅ 上传成功：{local_ckpt_path} → https://huggingface.co/{repo_id}")
