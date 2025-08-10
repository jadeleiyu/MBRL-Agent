# ==============================================
# File: scripts/mpc_rollouts.py
# ==============================================
"""
Collecting model predictive control (MPC) action trajectories for distillation fine-tuning of web agents.

Given an instruction (or a file of instructions), this script:
  1) Samples N synthetic trajectories of length T *inside the world model*.
  2) Scores each trajectory with the *value model* (LangAGI-Lab value adapter).
  3) Selects the best-scoring trajectory and emits per-step SFT examples (chat-formatted `text`).

Output format (JSONL; one line per step of the selected best trajectory):
  {"instruction": ..., "score": <float>, "step_index": i, "text": "<chat formatted sample>"}

Example
-------
python -m scripts.mpc_rollouts \
  --policy_ckpt ./checkpoints/sft-llama3.1-8b \
  --wm_adapter LangAGI-Lab/Meta-Llama-3.1-8B-Instruct-WM-webarena-16k-adapter \
  --value_adapter LangAGI-Lab/Meta-Llama-3.1-8B-Instruct-value-model-16k-qlora-adapter-v2 \
  --instructions_file ./data/instructions.txt \
  --out ./distilled_sft.jsonl --samples 8 --horizon 10 --temperature 0.7
"""
import os
import json
import argparse
from typing import List, Dict

import torch
from transformers import AutoTokenizer

from ..agents.policy import PolicyWithValue
from ..envs.wm_env import WMAWorldModelEnv, WMState
from ..agents.mpc_value import MPCValueModel


def build_obs(instruction: str, state: WMState) -> str:
    return (
        "You are a web agent. Read the current page DOM and output ONE atomic action (CLICK/TYPE/SELECT/NAVIGATE) in a strict schema.\n"
        f"URL: {state.url}\nDOM:\n{state.dom}\nInstruction: {instruction}\nAction:"
    )


def sample_dream(env: WMAWorldModelEnv, policy: PolicyWithValue, instruction: str, horizon: int, temperature: float, max_new_tokens: int) -> List[Dict]:
    state = WMState(url="https://example.com", dom="<body>Home</body>", step=0)
    traj: List[Dict] = []
    for t in range(horizon):
        obs = build_obs(instruction, state)
        # sample action from policy
        with torch.inference_mode():
            toks = policy.tokenizer([obs], return_tensors="pt").to(policy.model.pretrained_model.device)
            gen = policy.model.generate(
                **toks,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.9,
            )
            action = policy.tokenizer.decode(gen[0][toks.input_ids.shape[1]:], skip_special_tokens=True)
        # predict next state with world model
        pred = env.predict_next(state, instruction, action)
        next_state = WMState(url=pred.get("next_url", state.url), dom=pred.get("next_dom", state.dom), done=bool(pred.get("done", False)), step=state.step + 1)
        traj.append({
            "url": state.url,
            "dom": state.dom,
            "action": action,
            "next_url": next_state.url,
            "next_dom": next_state.dom,
            "done": next_state.done,
        })
        state = next_state
        if next_state.done:
            break
    return traj


def to_sft_rows(tokenizer: AutoTokenizer, instruction: str, trajectory: List[Dict], score: float):
    rows = []
    for i, st in enumerate(trajectory):
        messages = [
            {"role": "system", "content": "You are a web agent. Output exactly ONE atomic action in the CLICK/TYPE/SELECT/NAVIGATE schema given the page DOM."},
            {"role": "user", "content": f"Instruction: {instruction}\nCurrent DOM:\n{st['dom']}\nAction:"},
            {"role": "assistant", "content": st["action"]},
        ]
        text = tokenizer.apply_chat_template(messages, tokenize=False)
        rows.append({
            "instruction": instruction,
            "score": score,
            "step_index": i,
            "text": text,
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_ckpt", default=os.getenv("POLICY_CKPT", "./checkpoints/sft-llama3.1-8b"))
    ap.add_argument("--wm_base", default=os.getenv("WM_BASE", "meta-llama/Meta-Llama-3.1-8B-Instruct"))
    ap.add_argument("--wm_adapter", default=os.getenv("WM_ADAPTER", "LangAGI-Lab/Meta-Llama-3.1-8B-Instruct-WM-webarena-16k-adapter"))
    ap.add_argument("--value_base", default=os.getenv("VALUE_BASE", "meta-llama/Meta-Llama-3.1-8B-Instruct"))
    ap.add_argument("--value_adapter", default=os.getenv("VALUE_ADAPTER", "LangAGI-Lab/Meta-Llama-3.1-8B-Instruct-value-model-16k-qlora-adapter-v2"))

    ap.add_argument("--instruction", default=None, help="Single instruction string")
    ap.add_argument("--instructions_file", default=None, help="Path to a file with one instruction per line")

    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_new_tokens", type=int, default=64)

    ap.add_argument("--out", required=True, help="Output JSONL for distilled SFT rows")

    args = ap.parse_args()

    # Load models
    policy = PolicyWithValue(args.policy_ckpt)
    env = WMAWorldModelEnv(base_model=args.wm_base, adapter=args.wm_adapter)
    val = MPCValueModel(base_model=args.value_base, adapter=args.value_adapter)

    tok = policy.tokenizer

    # Build instruction list
    instructions: List[str] = []
    if args.instruction:
        instructions.append(args.instruction)
    if args.instructions_file:
        with open(args.instructions_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    instructions.append(line)

    assert instructions, "Provide --instruction or --instructions_file"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fout = open(args.out, "w", encoding="utf-8")

    for instr in instructions:
        # sample N trajectories
        candidates: List[List[Dict]] = []
        for _ in range(args.samples):
            traj = sample_dream(env, policy, instr, args.horizon, args.temperature, args.max_new_tokens)
            candidates.append(traj)

        # score with value model and choose best
        scored = []
        for traj in candidates:
            score = val.score(instr, traj)
            scored.append((score, traj))
        scored.sort(key=lambda x: x[0], reverse=True)
        best_score, best_traj = scored[0]

        # write SFT rows to file
        for row in to_sft_rows(tok, instr, best_traj, best_score):
            fout.write(json.dumps(row) + "\n")
        fout.flush()
        print(f"[distill] instr='{instr[:50]}...' | best_score={best_score:.3f} | steps={len(best_traj)}")

    fout.close()
    print(f"Saved distilled SFT rows to {args.out}")


if __name__ == "__main__":
    main()
