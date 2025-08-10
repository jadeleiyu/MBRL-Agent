# ==============================================
# File: scripts/run_sft.py
# ==============================================
import os
from datasets import concatenate_datasets
from ..data.datasets import load_agentinstruct, load_mind2web
from ..training.sft_train import run_sft
from ..config import TrainingArgs

if __name__ == "__main__":
    model_name = os.getenv("POLICY_BASE", "meta-llama/Meta-Llama-3.1-8B-Instruct")
    out_dir = os.getenv("SFT_OUT", "./checkpoints/sft-llama3.1-8b")

    ds1 = load_mind2web("train")
    ds2 = load_agentinstruct("train")
    ds = concatenate_datasets([ds1, ds2]).shuffle(seed=42).select(range(5000))  # keep it small; adjust as needed

    run_sft(model_name, out_dir, ds, TrainingArgs())
    print(f"SFT saved to {out_dir}")

    