import os
from typing import Literal

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)

from ..config import TrainingArgs


def make_sft_corpus(ds: Dataset, template: Literal["chat", "plain"] = "chat"):
    def _map(ex):
        if template == "chat":
            messages = [
                {"role": "system", "content": "You are a helpful web agent that outputs one atomic action per reply in a strict schema."},
                {"role": "user", "content": ex["instruction"]},
                {"role": "assistant", "content": "THINK, then output an action like CLICK(\"#selector\") or TYPE(\"#selector\", \"text\")."},
            ]
            return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}
        else:
            return {"text": f"Instruction: {ex['instruction']}\nAction: CLICK(\"#submit\")"}
    return _map


def run_sft(model_name: str, output_dir: str, ds: Dataset, targs: TrainingArgs):
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    processed = ds.map(make_sft_corpus(ds), remove_columns=ds.column_names)

    def tok(ex):
        return tokenizer(ex["text"], truncation=True, max_length=targs.max_seq_len)

    tokenized = processed.map(tok, batched=True)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=targs.sft_batch_size,
        gradient_accumulation_steps=targs.gradient_accumulation_steps,
        learning_rate=targs.sft_lr,
        num_train_epochs=targs.sft_epochs,
        logging_steps=10,
        save_steps=500,
        bf16=targs.bf16,
        report_to=["none"],
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(model=model, args=args, train_dataset=tokenized, data_collator=collator)
    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    