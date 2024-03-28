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
# from torch.utils.data import Dataset
from datasets import load_dataset
from datasets import Features
from datasets.arrow_dataset import Dataset
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
    pipeline,
)
from transformers import Trainer, GPTQConfig, deepspeed, set_seed
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from accelerate import Accelerator
from accelerate.utils import DistributedType
from trl import SFTTrainer

# from rouge_chinese import Rouge
# from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from modelscope.msdatasets import MsDataset

logger = logging.getLogger(__name__)



@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='../models/')


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: Optional[str] = field(default="./results")
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
    save_steps:int = field(default=100)
    logging_steps:int = field(default=10)
    learning_rate:float = field(default=2e-4)
    max_grad_norm:float = field(default=0.3)
    max_steps:int = field(default=100)
    warmup_ratio:float = field(default=0.03)
    lr_scheduler_type:str = field(default="constant")
    remove_unused_columns:bool = field(default=False)
    group_by_length:bool = field(default=True, metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },)


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_path,
        tokenizer,
        model_max_length,
        user_tokens=[195],
        assistant_tokens=[196],
    ):
        super(SupervisedDataset, self).__init__()
        self.data = json.load(open(data_path))
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.user_tokens = user_tokens
        self.assistant_tokens = assistant_tokens
        self.ignore_index = -100
        item = self.preprocessing(self.data[0])
        print("input:", self.tokenizer.decode(item["input_ids"]))
        labels = []
        for id_ in item["labels"]:
            if id_ == -100:
                continue

            labels.append(id_)
        print("label:", self.tokenizer.decode(labels))

    def __len__(self):
        return len(self.data)

    def preprocessing(self, example):
        input_ids = []
        labels = []

        for message in example["conversations"]:
            from_ = message["from"]
            value = message["value"]
            value_ids = self.tokenizer.encode(value)

            if from_ == "human":
                input_ids += self.user_tokens + value_ids
                labels += [self.tokenizer.eos_token_id] + [self.ignore_index] * len(
                    value_ids
                )
            else:
                input_ids += self.assistant_tokens + value_ids
                labels += [self.ignore_index] + value_ids
        input_ids.append(self.tokenizer.eos_token_id)
        labels.append(self.tokenizer.eos_token_id)
        input_ids = input_ids[: self.model_max_length]
        labels = labels[: self.model_max_length]
        input_ids += [self.tokenizer.pad_token_id] * (
            self.model_max_length - len(input_ids)
        )
        labels += [self.ignore_index] * (self.model_max_length - len(labels))
        input_ids = torch.LongTensor(input_ids)
        labels = torch.LongTensor(labels)
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

    def __getitem__(self, idx) -> Dict[str, torch.Tensor]:
        return self.preprocessing(self.data[idx])


def gen_batches_train():
    # https://plainenglish.io/community/fine-tuning-mistral-7b-model-with-your-custom-data-010eb6
    ds = MsDataset.load('AI-ModelScope/DISC-Law-SFT', subset_name='default', split='train', cache_dir='../inputs')
    total_samples = 10000
    val_pct = 0.1
    train_limit = int(total_samples * (1 - val_pct))
    counter = 0
    trainbatch = []

    for sample in iter(ds):
        if counter >= train_limit:
            break

        prompt = sample['input']
        response = sample['output']
        new_text_format = f'### Human: {prompt} ### Assistant: {response}'
    
        trainbatch.append({'text': new_text_format})
        counter += 1

    return trainbatch


def back():
    def read_jsonl(path):
        data = []
        with open(path, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    train_data_list = read_jsonl('./output_examples_train.jsonl')
    valid_data_list = read_jsonl('./output_examples_val.jsonl')

    tr_data_dict = {'text': [item['text'] for item in train_data_list]}
    val_data_dict = {'text': [item['text'] for item in valid_data_list]}

    train_dataset = Dataset.from_dict(tr_data_dict)
    valid_dataset = Dataset.from_dict(val_data_dict)


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print(training_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    # datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        # use_fast=False,
        encode_special_tokens=True,
        trust_remote_code=True,
        # model_max_length=training_args.model_max_length,
        # cache_dir=training_args.cache_dir,
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # dataset = SupervisedDataset(data_args.data_path, tokenizer, training_args.model_max_length)
    # dataset = load_dataset(path = '/root/share/datasets/openassistant-guanaco')
    # dataset = MsDataset.load('AI-ModelScope/DISC-Law-SFT', subset_name='default', split='train', cache_dir='../inputs')
    # ds =  MsDataset.load('Daxusell/china_law_set')


    # column_names = list(next(iter(dataset)).keys())
    # print('*'*20, column_names)

    # features = Features.from_dict(
    #     {
    #         "prompt": [
    #             {"role": {"dtype": "string", "_type": "Value"}, "content": {"dtype": "string", "_type": "Value"}}
    #         ],
    #         "response": [
    #             {"role": {"dtype": "string", "_type": "Value"}, "content": {"dtype": "string", "_type": "Value"}}
    #         ],
    #         "system": {"dtype": "string", "_type": "Value"},
    #         "tools": {"dtype": "string", "_type": "Value"},
    #     }
    # )
    # train_dataset = dataset.to_hf_dataset()  # 魔搭社区下载

    # def modelscope_to_df(examples):
    #     outputs = {"prompt": [], "response": [], "system": [], "tools": []}
    #     for i, message in enumerate(examples[]):

    train_dataset = Dataset.from_list(gen_batches_train())
    print(train_dataset[0])
 

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
        model_args.model_name_or_path,
        # quantization_config=bnb_config,
        load_in_8bit=True,
        trust_remote_code=True,
        # cache_dir=training_args.cache_dir,
    )
    model.config.use_cache = False
    # response, history = model.chat(tokenizer, "please provide three suggestions about time management", history=[])
    # print(response)    
    
    
    if training_args.use_lora:  
        from peft import LoraConfig, TaskType, get_peft_model

        peft_config = LoraConfig(
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                # "o_proj"
                ],
            r=64,
            bias="none",
            task_type="CAUSAL_LM"
        )       
        # model.enable_input_require_grads()
        # model = get_peft_model(model, peft_config)
        # model.print_trainable_parameters()
   
    # trainer = transformers.Trainer(
    #     model=model, args=training_args, train_dataset=train_dataset["train"], tokenizer=tokenizer
    # )
    # trainer.train()
    # trainer.save_state()
    # trainer.save_model(output_dir=training_args.output_dir)  

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=training_args.model_max_length,
        tokenizer=tokenizer,
        args=training_args,
    )
    trainer.train()

    # trainer.save_model(training_args.output_dir)
    trainer.model.save_pretrained("internlm-2-7b-layer")

    if False:
        model = AutoPeftModelForCausalLM.from_pretrained(output_dir, device_map="auto", torch_dtype=torch.bfloat16)
        model = model.merge_and_unload()

        output_merged_dir = os.path.join(script_args.output_dir, "final_merged_checkpoint")
        model.save_pretrained(output_merged_dir, safe_serialization=True)


def inference():
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        low_cpu_mem_usage=True,
        return_dict=True,
        torch_dtype=torch.float16,
        device_map=device_map,
    )
    model = PeftModel.from_pretrained(base_model, new_model)
    model = model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    prompt = "What is a large language model?"
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result[0]['generated_text'])


if __name__ == "__main__":
    train()
