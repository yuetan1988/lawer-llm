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
    prepare_model_for_int8_training,
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
from trl import SFTTrainer

logger = logging.getLogger(__name__)
