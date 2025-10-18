import json
from vllm import LLM, SamplingParams
from rllm.environments.base.multi_turn_env import MultiTurnEnvironment


class LLMDrivenEnvironment(MultiTurnEnvironment):
    def __init__(self, task: dict, model_path: str, max_turns: int = 3, temperature: float = 0.2, top_p: float = 0.9):
        super().__init__(task=task, max_turns=max_turns)
        self.model_path = model_path
        self.llm = LLM(model=model_path, tensor_parallel_size=1)
        self.sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=512)

    def _run_llm(self, prompt: str) -> str:
        outputs = self.llm.generate(prompt, self.sampling_params)
        return outputs[0].outputs[0].text.strip()

    def get_reward_and_next_obs(self, task: dict, action: str):
        current_q = ""
        if "questions" in task and self.current_turn < len(task["questions"]):
            current_q = task["questions"][self.current_turn]
        prompt = f"""
                You are an evaluator for an interactive reasoning task.

                Current Question: {current_q}
                Agent Response: {action}

                Now:
                1. Evaluate how correct and coherent the answer is.
                2. Give a score between 0.0 and 1.0 (float).
                3. Propose the next question or follow-up step.

                Respond strictly in JSON format:
                {{
                "score": <float>,
                "next_question": <string>
                }}
                """
        llm_output = self._run_llm(prompt)
        # Try to extract JSON
        try:
            data = json.loads(llm_output)
            reward = float(data.get("score", 0))
            next_q = data.get("next_question", "")
        except Exception:
            # fallback if parsing fails
            reward = 0.0
            next_q = "Could not parse JSON."

        next_obs = {"question": next_q}
        return reward, next_obs

    @staticmethod
    def from_dict(env_args: dict):
        task = env_args.get("task", {})
        model_path = env_args.get("model_path")
        max_turns = env_args.get("max_turns", 3)
        return LLMDrivenEnvironment(task=task, model_path=model_path, max_turns=max_turns)
