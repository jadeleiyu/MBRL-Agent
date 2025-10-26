# ==============================================
# File: vanilla_policy.py
# ==============================================
import re, os
from transformers import AutoTokenizer

os.environ["VLLM_CONFIGURE_LOGGING"] = "0"   # set this *before* importing vllm
os.environ["VLLM_LOGGING_LEVEL"]    = "WARNING"  # or "ERROR"
from vllm import LLM, SamplingParams

from mbrl_agent.agents.prompt_constructor import CoTPromptConstructor
from mbrl_agent.envs.browser_env.actions import (
    ActionParsingError,
    create_id_based_action,
)


class VanillaPolicy:
    """
    Vanilla web agent policy (for inference only).
    """

    def __init__(self, args):

        agent_model_name = f"{args.agent_model_name}-{args.env}"
        self.model = LLM(
            model=agent_model_name,
            trust_remote_code=True,
            tensor_parallel_size=args.agent_tensor_parallel_size,
            dtype=args.torch_dtype,
        )
        tokenizer = AutoTokenizer.from_pretrained(agent_model_name)
        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.agent_max_new_tokens,
        )
        self.prompt_constructor = CoTPromptConstructor(
            args.agent_prompt_path, tokenizer, args.max_obs_length
        ) 
        self.env_type = args.env_type # "dream", "real"


    def act(self, batch_trajs, batch_tasks):
        batch_input_text = []
        for i in range(len(batch_trajs)):
            input_text = self.prompt_constructor.construct(
                trajectory=batch_trajs[i], intent=batch_tasks[i]['objective']
            )
            batch_input_text.append(input_text)
        batch_outputs = self.model.generate(
            batch_input_text, self.sampling_params, use_tqdm=False
        )
        
        batch_actions, batch_is_valild_act, batch_responses = [], [], []
        for out in batch_outputs:
            if self.env_type == 'dream':
                try:
                    action = self.prompt_constructor._extract_action(out.outputs[0].text)
                    batch_is_valild_act.append(True)
                    batch_actions.append(action)
                    batch_responses.append(out.outputs[0].text)
                except ActionParsingError as e:
                    batch_actions.append(out.outputs[0].text)
                    batch_is_valild_act.append(False)
                    batch_responses.append(out.outputs[0].text)
                
            else:
                action_text = batch_outputs[0].outputs[0].text
                parsed_response = self.prompt_constructor.extract_action(
                    action_text
                )
                action = create_id_based_action(parsed_response)
                batch_actions.append(action)
                batch_responses.append(action_text)
                
        return batch_responses, batch_actions, batch_is_valild_act
    
    def is_stop_act(self, action):
        return re.search(r"stop ?\[(.+)\]", action)
        



        
