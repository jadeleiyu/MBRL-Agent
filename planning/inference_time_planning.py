"""model-based web-agent with inference time planning."""

import argparse
import json
import torch
from typing import List, Dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from mbrl import WebWorldModel, VanillaPolicy
from mbrl.envs.webvoyager_env import WV_SYSTEM_PROMPT, WV_INIT_USER_PROMPT, WebVoyagerEnv, driver_config


class ValueFunction:
    """Value function for estimating rewards of states"""
    
    def __init__(self, model_path: str = "path/to/value_function_checkpoint"):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        
    def estimate_reward(self, 
                       instruction: str,
                       current_obs: str,
                       action: str,
                       predicted_next_obs: str) -> float:
        """
        Estimate the reward/value of taking an action.
        Returns a score between 0 and 1.
        """
        prompt = self._construct_prompt(instruction, current_obs, action, predicted_next_obs)
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=256,
                temperature=0.3,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True)
        
        # Extract score from response
        score = self._extract_score(response)
        return score
    
    def _construct_prompt(self, instruction: str, current_obs: str, action: str, next_obs: str) -> str:
        """Construct prompt for value function"""
        prompt = f"""Evaluate how well this action contributes to the task goal.

User's objective: {instruction}
Current state: {current_obs}
Action: {action}
Predicted next state: {next_obs}

Provide a score between 0 and 1 for how well this action moves towards the task completion.
[Rationale]"""
        return prompt
    
    def _extract_score(self, response: str) -> float:
        """Extract score from value function response"""
        import re
        
        # Look for score patterns like [Score] 0.8 or similar
        patterns = [
            r'\[Score\]\s*([0-9]*\.?[0-9]+)',
            r'score:\s*([0-9]*\.?[0-9]+)',
            r'([0-9]*\.?[0-9]+)\s*\/\s*1'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                try:
                    score = float(match.group(1))
                    return min(max(score, 0.0), 1.0)  # Clamp between 0 and 1
                except:
                    pass
        
        # Default score if extraction fails
        return 0.5


class WMAPolicy:
    """World Model Augmented Policy for web navigation"""
    
    def __init__(self, 
                 policy_model,
                 world_model_path: str = None,
                 value_function_path: str = None,
                 k_samples: int = 5):
        """
        Initialize WMA Policy.
        
        Args:
            policy_model: Base policy model for generating action candidates
            world_model_path: Path to world model checkpoint
            value_function_path: Path to value function checkpoint
            k_samples: Number of action candidates to sample
        """
        self.policy_model = policy_model
        self.world_model = WebWorldModel(world_model_path) if world_model_path else None
        self.value_function = ValueFunction(value_function_path) if value_function_path else None
        self.k_samples = k_samples
        
    def act_with_planning(self, 
                         messages: List[Dict],
                         instruction: str,
                         current_obs: str) -> str:
        """
        Generate action using world model planning.
        
        Args:
            messages: Conversation history
            instruction: Task instruction
            current_obs: Current observation (accessibility tree)
            
        Returns:
            Selected action string
        """
        # Sample k action candidates from policy model
        action_candidates = self._sample_action_candidates(messages)
        
        if not self.world_model or not self.value_function or len(action_candidates) == 1:
            # Fallback to vanilla policy if models not available or only one candidate
            return action_candidates[0]
        
        # Simulate outcomes and score each candidate
        action_scores = []
        for action in action_candidates:
            # Predict next observation using world model
            predicted_next_obs = self.world_model.predict_next_observation(
                instruction, current_obs, action
            )
            
            # Estimate reward using value function
            reward = self.value_function.estimate_reward(
                instruction, current_obs, action, predicted_next_obs
            )
            
            action_scores.append((action, reward, predicted_next_obs))
        
        # Select action with highest reward
        best_action, best_score, best_next_obs = max(action_scores, key=lambda x: x[1])
        
        print(f"\n=== WMA Planning Results ===")
        print(f"Sampled {len(action_candidates)} candidates")
        print(f"Best action score: {best_score:.3f}")
        print(f"Predicted outcome: {best_next_obs[:200]}...")
        print("===========================\n")
        
        return best_action
    
    def _sample_action_candidates(self, messages: List[Dict]) -> List[str]:
        """Sample k diverse action candidates from policy model"""
        candidates = []
        
        for i in range(self.k_samples):
            # Use temperature sampling for diversity
            action = self.policy_model.act(messages, temperature=0.8)
            candidates.append(action)
        
        # Deduplicate while preserving order
        seen = set()
        unique_candidates = []
        for action in candidates:
            # Extract just the action part for comparison
            action_key = self._extract_action_key(action)
            if action_key not in seen:
                seen.add(action_key)
                unique_candidates.append(action)
        
        return unique_candidates[:self.k_samples]
    
    def _extract_action_key(self, action_text: str) -> str:
        """Extract the core action for deduplication"""
        import re
        
        # Look for action patterns like click[123], type[456], etc.
        patterns = [
            r'(click\s*\[[^\]]+\])',
            r'(type\s*\[[^\]]+\]\s*\[[^\]]+\])',
            r'(scroll\s*\[[^\]]+\])',
            r'(hover\s*\[[^\]]+\])',
            r'(stop\s*\[[^\]]+\])',
            r'(new_tab)',
            r'(close_tab)',
            r'(go_back)',
            r'(go_forward)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, action_text, re.IGNORECASE)
            if match:
                return match.group(1).lower()
        
        return action_text.lower()



def main(args):
    # Load the base policy model
    base_policy = VanillaPolicy(args)
    
    # Create WMA policy with world model and value function
    wma_policy = WMAPolicy(
        policy_model=base_policy,
        world_model_path=args.world_model_path,
        value_function_path=args.value_function_path,
        k_samples=args.k_samples
    )
    
    # Create web driver options
    driver_options = driver_config(args)
    
    # Load webvoyager tasks
    tasks = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            tasks.append(json.loads(line))
    
    success_count = 0
    total_count = 0
    
    for task_id in range(len(tasks)):
        task = tasks[task_id]
        env = WebVoyagerEnv(driver_options, task, args)
        
        ac_tree, obs_info = env.get_webarena_accessibility_tree()
        messages = [
            {"role": "system", "content": WV_SYSTEM_PROMPT},
            {"role": "user", "content": WV_INIT_USER_PROMPT.format(
                instruction=task['instruction'], web=task['web'], ac_tree=ac_tree
            )},
        ]
        
        instruction = task['instruction']
        trajectory = []
        
        it = 0
        while it < args.max_iter:
            ac_tree, obs_info = env.get_webarena_accessibility_tree()
            
            if it > 0:
                messages.append({
                    "role": "user", 
                    "content": f"Please analyze the accessibility tree and give the Thought and Action.\n{ac_tree}"
                })
            
            # Use WMA policy with planning
            if args.use_wma:
                action_text = wma_policy.act_with_planning(
                    messages=messages,
                    instruction=instruction,
                    current_obs=ac_tree
                )
            else:
                # Fallback to vanilla policy
                action_text = base_policy.act(messages)
            
            messages.append({'role': 'assistant', 'content': action_text})
            trajectory.append({
                'observation': ac_tree,
                'action': action_text,
                'iteration': it
            })
            
            # Execute action in environment
            done = env.step(action_text, obs_info)
            
            if done or 'stop' in action_text.lower():
                break
            
            it += 1
        
        # Evaluate success (simplified - you may want to implement proper evaluation)
        success = env.evaluate()  # Assuming env has an evaluate method
        if success:
            success_count += 1
        total_count += 1
        
        print(f"Task {task_id}: {'SUCCESS' if success else 'FAIL'}")
        print(f"Current success rate: {success_count}/{total_count} = {success_count/total_count:.2%}")
        
        # Save trajectory if needed
        if args.save_trajectories:
            with open(f"trajectories/task_{task_id}.json", 'w') as f:
                json.dump({
                    'task': task,
                    'trajectory': trajectory,
                    'success': success
                }, f, indent=2)
    
    print(f"\nFinal Results:")
    print(f"Success Rate: {success_count}/{total_count} = {success_count/total_count:.2%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Environment arguments
    parser.add_argument("--test_file", type=str, required=True, help="Path to test tasks file")
    parser.add_argument("--max_iter", type=int, default=15, help="Maximum iterations per task")
    
    # WMA arguments
    parser.add_argument("--use_wma", action="store_true", help="Use WMA planning")
    parser.add_argument("--world_model_path", type=str, default=None, help="Path to world model checkpoint")
    parser.add_argument("--value_function_path", type=str, default=None, help="Path to value function checkpoint")
    parser.add_argument("--k_samples", type=int, default=5, help="Number of action candidates to sample")
    
    # Other arguments
    parser.add_argument("--save_trajectories", action="store_true", help="Save trajectories to disk")
    
    args = parser.parse_args()
    main(args)