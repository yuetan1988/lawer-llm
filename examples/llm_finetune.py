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
# from rouge_chinese import Rouge
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

logger = logging.getLogger(__name__)



ds = load_dataset(path = '/root/share/datasets/openassistant-guanaco')

print(ds['train'])
print(ds['train'][0]['text'])

