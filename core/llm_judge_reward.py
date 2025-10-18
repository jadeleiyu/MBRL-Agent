"""
LLM-as-Judge Reward System
Uses LLM as a judge to evaluate agent performance and provide rewards
"""

import json
from typing import Dict, Any, Optional
from vllm import LLM, SamplingParams
from rllm.rewards.reward_fn import RewardFunction, RewardOutput
from rllm.rewards.reward_types import RewardInput


class LLMJudgeRewardFunction(RewardFunction):
    """LLM as judge reward function"""

    def __init__(
        self,
        model_path: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        max_tokens: int = 512,
        judge_prompt_template: Optional[str] = None,
        llm_instance: Optional[LLM] = None
    ):
        """
        Initialize LLM judge

        Args:
            model_path: LLM model path
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            max_tokens: Maximum tokens to generate
            judge_prompt_template: Judge prompt template
            llm_instance: Shared LLM instance (optional)
        """
        self.model_path = model_path

        # Use shared LLM instance if provided, otherwise create new one
        if llm_instance is not None:
            self.llm = llm_instance
        else:
            self.llm = LLM(model=model_path, tensor_parallel_size=1)

        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )

        # Default judge prompt template
        self.judge_prompt_template = judge_prompt_template or """
You are a professional judge responsible for evaluating agent performance in tasks.

Task Description: {task_description}
Current Question: {current_question}
Agent Response: {agent_response}

Please evaluate based on the following criteria:
1. Answer correctness and accuracy (0-1 points)
2. Answer relevance and completeness (0-1 points)
3. Language clarity and fluency (0-1 points)

Please respond strictly in JSON format:
{{
    "overall_score": <overall score, float between 0.0-1.0>,
    "correctness": <correctness score, float between 0.0-1.0>,
    "relevance": <relevance score, float between 0.0-1.0>,
    "clarity": <clarity score, float between 0.0-1.0>,
    "feedback": <specific feedback and suggestions>
}}
"""

    def __call__(self, reward_input: RewardInput) -> RewardOutput:
        """
        Calculate reward value

        Args:
            reward_input: Reward input containing task info and action

        Returns:
            RewardOutput: Reward output
        """
        task_info = reward_input.task_info
        action = reward_input.action

        # Build judge prompt
        prompt = self.judge_prompt_template.format(
            task_description=task_info.get("instruction", ""),
            current_question=task_info.get("question", ""),
            agent_response=action
        )

        try:
            # Call LLM for judgment
            outputs = self.llm.generate(prompt, self.sampling_params)
            llm_output = outputs[0].outputs[0].text.strip()

            # Parse LLM output
            evaluation = self._parse_llm_output(llm_output)
            overall_score = evaluation.get("overall_score", 0.0)

            return RewardOutput(
                reward=overall_score,
                metadata={
                    "llm_judge_output": llm_output,
                    "detailed_scores": evaluation,
                    "judge_model": self.model_path
                }
            )

        except Exception as e:
            # Return default reward if judgment fails
            return RewardOutput(
                reward=0.0,
                metadata={
                    "error": str(e),
                    "judge_model": self.model_path
                }
            )

    def _parse_llm_output(self, llm_output: str) -> Dict[str, Any]:
        """
        Parse LLM output to extract scoring information

        Args:
            llm_output: LLM output text

        Returns:
            Parsed scoring dictionary
        """
        try:
            # Try to extract JSON part
            json_start = llm_output.find("{")
            json_end = llm_output.rfind("}") + 1

            if json_start >= 0 and json_end > json_start:
                json_str = llm_output[json_start:json_end]
                data = json.loads(json_str)

                # Validate score ranges
                for key in ["overall_score", "correctness", "relevance", "clarity"]:
                    if key in data:
                        score = float(data[key])
                        data[key] = max(0.0, min(1.0, score))  # Ensure in 0-1 range

                return data
            else:
                # Return default values if no JSON found
                return {
                    "overall_score": 0.0,
                    "correctness": 0.0,
                    "relevance": 0.0,
                    "clarity": 0.0,
                    "feedback": "Unable to parse LLM output"
                }

        except Exception as e:
            return {
                "overall_score": 0.0,
                "correctness": 0.0,
                "relevance": 0.0,
                "clarity": 0.0,
                "feedback": f"Parsing error: {str(e)}"
            }


# Convenience function
def create_llm_judge_reward(
    model_path: str,
    temperature: float = 0.2,
    top_p: float = 0.9,
    max_tokens: int = 512,
    llm_instance: Optional[LLM] = None
) -> LLMJudgeRewardFunction:
    """
    Convenience function to create LLM judge reward function

    Args:
        model_path: Model path
        temperature: Sampling temperature
        top_p: Top-p sampling parameter
        max_tokens: Maximum tokens to generate
        llm_instance: Shared LLM instance (optional)

    Returns:
        LLMJudgeRewardFunction instance
    """
    return LLMJudgeRewardFunction(
        model_path=model_path,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        llm_instance=llm_instance
    )