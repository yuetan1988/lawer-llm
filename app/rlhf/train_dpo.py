import gc
import os
import logging
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import bitsandbytes as bnb
import torch
import transformers
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
)
from transformers.generation.utils import GenerationConfig
from trl import DPOTrainer


@dataclass
class DataArguments:
    data_path: str = field(
        default="../../inputs/hh-rlhf2/train_hhrlhf.json",
        metadata={"help": "Path to the training data."},
    )
    prompt_length: int = field(default=128)
    source_length: int = field(default=512)
    target_length: int = field(default=512)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="../../models/")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default="../../outputs/result")
    cache_dir: Optional[str] = field(default=None)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=4)
    optim: str = field(default="paged_adamw_32bit")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    use_lora: bool = field(default=True)
    save_steps: int = field(default=100)
    logging_steps: int = field(default=50)
    learning_rate: float = field(default=2e-4)
    max_grad_norm: float = field(default=0.3)
    max_steps: int = field(default=1000)  # TEMP
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="constant")
    remove_unused_columns: bool = field(default=False)
    group_by_length: bool = field(
        default=True,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    use_deepspeed: bool = field(default=False)
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )


def build_dpo_data(data_args):
    datasets = load_dataset(
        "json",
        data_files={"train": data_args.data_path},
        # cache_dir=cache_dir,
    )
    train_dataset = datasets["train"]

    train_dataset = train_dataset.map(preprocess).shuffle()
    return train_dataset


def build_prompt(prompt_and_response):
    # TODO
    search_term = "Assistant:"
    search_term_idx = prompt_and_response.rfind(search_term)
    assert (
        search_term_idx != -1
    ), f"Prompt and response does not contain '{search_term}'"
    return prompt_and_response[: search_term_idx + len(search_term)]


def preprocess(sample):
    prompt = build_prompt(sample["chosen"])
    return {
        "prompt": prompt.replace("\n\nHuman: ", "<reserved_106>").replace(
            "\n\nAssistant: ", "<reserved_107>"
        ),
        "chosen": sample["chosen"][len(prompt) :],
        "rejected": sample["rejected"][len(prompt) :],
    }


def build_model(model_args, training_args):
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16,
        load_in_8bit=True,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if training_args.use_lora:
        from peft import LoraConfig, get_peft_model

        LORA_R = 32
        LORA_DROPOUT = 0.05
        TARGET_MODULES = [
            "o_proj",
            "W_pack",
        ]

        config = LoraConfig(
            r=LORA_R,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    if training_args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]
    return model


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(training_args)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        # use_fast=False,
        encode_special_tokens=True,
        trust_remote_code=True,
        # model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )
    tokenizer.pad_token = tokenizer.eos_token

    with training_args.main_process_first(desc="loading and tokenization"):
        train_dataset = build_dpo_data(data_args)

    model = build_model(model_args, training_args)
    model_ref = None

    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=0.1,
        train_dataset=train_dataset,
        # eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=data_args.source_length,
        max_target_length=data_args.target_length,
        max_prompt_length=data_args.prompt_length,
        generate_during_eval=False,
    )
    trainer.train()


if __name__ == "__main__":
    train()
