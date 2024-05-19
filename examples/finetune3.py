import json
import logging
import os
import pathlib
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import bitsandbytes as bnb
import numpy as np
import pandas as pd
import torch
import transformers
from accelerate import Accelerator
from accelerate.utils import DistributedType

# from torch.utils.data import Dataset
from datasets import Features, load_dataset
from datasets.arrow_dataset import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus

# from rouge_chinese import Rouge
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from modelscope.msdatasets import MsDataset
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_config,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
)
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    GPTQConfig,
    HfArgumentParser,
    LlamaForCausalLM,
    LlamaTokenizer,
    Seq2SeqTrainingArguments,
    Trainer,
    deepspeed,
    pipeline,
    set_seed,
)
from transformers.training_args import TrainingArguments
# from trl import SFTTrainer

logger = logging.getLogger(__name__)


PROMPT = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)
IGNORE_INDEX = -100


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)


@dataclass
class DataArguments:
    data_path: str = field(
        default="/root/lawer-llm/inputs/train_data.json",
        metadata={"help": "Path to the training data."},
    )
    source_length: int = field(default=512)
    target_length: int = field(default=512)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default=None)
    cache_dir: Optional[str] = field(default=None)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=4)
    optim: str = field(default="paged_adamw_32bit")  #   # "adamw_torch"
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
    # max_steps: int = field(default=1000)
    warmup_ratio: float = field(default=0.03)
    lr_scheduler_type: str = field(default="constant")
    remove_unused_columns: bool = field(default=False)
    group_by_length: bool = field(
        default=False,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    use_deepspeed: bool = field(default=False)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(training_args)


if __name__ == '__main__':
    train()
