# ==============================================
# File: vanilla_policy.py
# ==============================================
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class VanillaPolicy:
    """
    Vanilla web agent policy (for inference only).
    """

    def __init__(self, args):

        self.model = LLM(args.model_name, device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.sampling_params = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            max_tokens=args.max_tokens,
        )   

    def act(self, messages) -> str:

        input_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        outputs = self.model.generate([input_text], self.sampling_params)
        action_text = outputs[0].outputs[0].text

        return action_text


        
