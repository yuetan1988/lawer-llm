import os
import re
import sys
import json
import time
import logging
import pathlib
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from dataclasses import dataclass, field
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import bitsandbytes as bnb
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    prepare_model_for_int8_training,
    set_peft_model_state_dict,
)
import transformers
from transformers.training_args import TrainingArguments
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
)
from transformers import Trainer, GPTQConfig, deepspeed, set_seed
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from accelerate import Accelerator
from accelerate.utils import DistributedType
from trl import SFTTrainer

# from rouge_chinese import Rouge
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

logger = logging.getLogger(__name__)



train_dataset = load_dataset(path = '/root/share/datasets/openassistant-guanaco')

print(train_dataset['train'])
print(train_dataset['train'][0]['text'])



model_name_or_path = '../models/'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_in_8bit=False,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)



model = AutoModelForCausalLM.from_pretrained(
    model_name_or_path,
    # quantization_config=bnb_config,
    load_in_8bit=True,
    trust_remote_code=True
)
model.config.use_cache = False


tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, encode_special_tokens=True, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# chat
# response, history = model.chat(tokenizer, "please provide three suggestions about time management", history=[])
# print(response)


lora_alpha = 16
lora_dropout = 0.1
lora_r = 64

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        # "o_proj"
        ],
    r=lora_r,
    bias="none",
    task_type="CAUSAL_LM"
)

output_dir = "./results"
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
optim = "paged_adamw_32bit"
save_steps = 100
logging_steps = 10
learning_rate = 2e-4
max_grad_norm = 0.3
max_steps = 100
warmup_ratio = 0.03
lr_scheduler_type = "constant"

training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=1,
    # optim=optim,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=True,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=True,
    lr_scheduler_type=lr_scheduler_type,
)



max_seq_length = 512

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset["train"],
    peft_config=peft_config,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
)

trainer.train()
