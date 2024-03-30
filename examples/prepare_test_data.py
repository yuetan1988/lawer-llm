import copy
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from datasets import load_dataset
from functools import partial
import transformers
from transformers import AutoTokenizer


IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
}
prompt_input, prompt_no_input = PROMPT_DICT["prompt_input"], PROMPT_DICT["prompt_no_input"]


def _tokenize_fn(strings, tokenizer):
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]
    input_ids = labels = [tokenized.input_ids[0]
                          for tokenized in tokenized_list]
    ne_pad_token_id = IGNORE_INDEX if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(ne_pad_token_id).sum().item() for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(
        sources,
        targets,
        tokenizer: transformers.PreTrainedTokenizer,
):
    """Preprocess the data by tokenizing."""
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [_tokenize_fn(
        strings, tokenizer) for strings in (examples, sources)]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def generate_sources_targets(examples, tokenizer: transformers.PreTrainedTokenizer):
    print(len(examples))
    
  
    if 'instruction' in examples:
        ins_data = examples['instruction']
        input_data = examples['input']
    else:
        ins_data = examples['input']
        input_data = [''] * len(examples)
    output = examples['output']

    len_ = len(ins_data)
    print(len_, len(examples), len(input_data), len(output))

    print(examples[0][:10])
    print(examples[1][:10])
    print(examples[2][:10])
    print(examples[3][:10])
    print(examples[4][:10])
    assert 0==1


    sources = [prompt_input.format_map({'instruction': ins_data[i], 'input': input_data[i]}) if input_data[i] != "" 
    else prompt_no_input.format_map({'instruction': ins_data[i]})
    for i in range(len_)]
    sources = [i[:data_args.source_length] for i in sources]
    targets = [
        f"{example[:data_args.target_length-1]}{tokenizer.eos_token}" for example in output]

    input_output = preprocess(
        sources=sources, targets=targets, tokenizer=tokenizer)
    examples['input_ids'] = input_output['input_ids']
    examples['labels'] = input_output['labels']
    return examples


datasets = load_dataset(
    'json',
    data_files={'train' :'../inputs/train_data.json'},
    # cache_dir=cache_dir,
)

train_dataset = datasets['train']
print(train_dataset)


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default='../models/')



@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    source_length: int = field(default=512)
    target_length: int = field(default=512)


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



parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
model_args, data_args, training_args = parser.parse_args_into_dataclasses()


tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        # use_fast=False,
        encode_special_tokens=True,
        trust_remote_code=True,
        # model_max_length=training_args.model_max_length,
        cache_dir=training_args.cache_dir,
    )
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


train_dataset = train_dataset.map(
    function=partial(generate_sources_targets, tokenizer=tokenizer),
    batched=True,
    # remove_columns=["text"],
    desc="Running tokenizer on train dataset",
    num_proc=20
).shuffle()
