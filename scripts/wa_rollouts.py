# ==============================================
# File: mbrl_agent/scripts/wa_rollouts.py
# ==============================================
"""
Collect agent trajectories in a WebArena sandbox and save to a JSONL replay buffer.

Schema (one episode per line):
{
  "instruction": str,
  "steps": [
    {"url": str, "dom": str, "action": str, "next_url": str, "next_dom": str, "done": bool}
  ]
}

We rely on a pluggable environment adapter so you can bring your own WebArena wrapper.
The adapter must implement:
  - reset(task: dict) -> state where state has at least {"url": str, "dom": str}
  - step(state, action: str) -> (next_state, done: bool, info: dict)

Example
-------
python -m scripts.collect_rollouts \
  --policy_ckpt ./checkpoints/sft-llama3.1-8b \
  --out ./rollouts/webarena_buffer.jsonl \
  --env-module your_pkg.webarena_adapter --env-class WebArenaEnvAdapter \
  --episodes 200 --horizon 10
"""
import os
import json
import argparse
import importlib
from typing import List

from ..agents.policy import PolicyWithValue
from ..envs.wm_env import WMState


def build_observation(instruction: str, state_dom: str, url: str) -> str:
    return (
        "You are a web agent. Read the current page DOM and output ONE atomic action (CLICK/TYPE/SELECT/NAVIGATE) in a strict schema.\n"
        f"URL: {url}\nDOM:\n{state_dom}\nInstruction: {instruction}\nAction:"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy_ckpt", default=os.getenv("POLICY_CKPT", "./checkpoints/sft-llama3.1-8b"))
    ap.add_argument("--env-module", required=True)
    ap.add_argument("--env-class", required=True)
    ap.add_argument("--out", required=True, help="Output JSONL file for replay buffer")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--max_new_tokens", type=int, default=64)

    args = ap.parse_args()

    # Load environment adapter
    mod = importlib.import_module(args.env_module)
    EnvCls = getattr(mod, args.env_class)
    env = EnvCls()

    # Load policy (generation only)
    policy = PolicyWithValue(args.policy_ckpt)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fout = open(args.out, "w", encoding="utf-8")

    for epi in range(args.episodes):
        # You can customize task creation by adding fields the adapter expects
        instruction = f"Episode {epi}: complete the goal on the website."
        task = {"instruction": instruction, "url": "https://example.com"}
        s = env.reset(task)
        if not isinstance(s, WMState):
            s = WMState(url=s.get("url", ""), dom=s.get("dom", ""), step=0)

        steps: List[dict] = []
        for t in range(args.horizon):
            obs = build_observation(instruction, s.dom, s.url)
            action = policy.act([obs], max_new_tokens=args.max_new_tokens)[0]
            ns, done, info = env.step(s, action)
            if not isinstance(ns, WMState):
                ns = WMState(url=ns.get("url", s.url), dom=ns.get("dom", s.dom), step=s.step + 1, done=bool(ns.get("done", False)))

            steps.append({
                "url": s.url,
                "dom": s.dom,
                "action": action,
                "next_url": ns.url,
                "next_dom": ns.dom,
                "done": bool(done),
            })
            s = ns
            if done:
                break

        episode = {"instruction": instruction, "steps": steps}
        fout.write(json.dumps(episode) + "\n")
        fout.flush()
        print(f"[collect] episode={epi} steps={len(steps)}")

    fout.close()
    print(f"Saved replay buffer to {args.out}")


if __name__ == "__main__":
    main()
