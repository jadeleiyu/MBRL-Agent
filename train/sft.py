from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, Mxfp4Config
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model

from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
)
import os
from dataclasses import dataclass

"""
accelerate launch \
    --config_file configs/zero3.yaml \
    sft.py \
    --config configs/sft_lora.yaml \
    --model_name_or_path openai/gpt-oss-20b \
    --packing true packing_strategy wrapped \
    --run_name gpt-oss-20b-nnetnav-wa-lora \
    --attn_implementation kernels-community/vllm-flash-attn3

accelerate launch \
    sft.py \
    --config configs/sft_lora.yaml \
    --model_name_or_path openai/gpt-oss-20b \
    --packing true packing_strategy wrapped \
    --run_name gpt-oss-20b-nnetnav-wa-lora \
    --attn_implementation kernels-community/vllm-flash-attn3
"""

@dataclass
class ExtraArgs:
    n_train_examples: int = 10000


def main(script_args, training_args, model_args, extra_args):

    dataset = load_dataset(script_args.dataset_name, split="train")
    dataset = dataset.remove_columns(['prompt', 'output'])
    if extra_args.n_train_examples > 0:
        dataset = dataset.select(range(extra_args.n_train_examples))

    quantization_config = Mxfp4Config(dequantize=True)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        quantization_config=quantization_config,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )
    
    model_name_short, ds_name_short = model_args.model_name_or_path.split('/')[-1], script_args.dataset_name.split('/')[-1]
    output_dir = os.path.join(training_args.output_dir, f"{model_name_short}_{ds_name_short}_{extra_args.n_train_examples}_sft_lora_{model_args.lora_r}")
    training_args.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.train()
    trainer.save_model(training_args.output_dir)
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(dataset_name=script_args.dataset_name)



if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, ExtraArgs))
    script_args, training_args, model_args, extra_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args, extra_args)




    




