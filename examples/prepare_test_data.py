import copy
import logging
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from datasets import load_dataset
from functools import partial
import transformers
from transformers import AutoTokenizer
from transformers import DataCollatorForSeq2Seq, Trainer


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
    # 注意examples的结构, 包含多个keys, 也就是df的列信息. 而其中的instruction/response等长度则是行信息    

    if 'instruction' in examples:
        ins_data = examples['instruction']
        input_data = examples['input']
    else:
        ins_data = examples['input']
        input_data = [''] * len(ins_data)
    output = examples['output']

    len_ = len(ins_data)
    # print(len_, len(examples), len(input_data), len(output), 'instruction' in examples)
    # assert 0==1

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
# print(train_dataset)
# print(train_dataset[0])
# print('-' * 50)
# print(train_dataset[1])



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
    use_deepspeed: bool = field(default=False)


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
    remove_columns=datasets["train"].column_names,
    desc="Running tokenizer on train dataset",
    num_proc=20
).shuffle()

# train_dataset = train_dataset.remove_columns(datasets["train"].column_names)

print(train_dataset[0])



def build_model(model_args: ModelArguments, training_args: TrainingArguments, data_args: DataArguments) -> tuple:

    if training_args.use_deepspeed:

        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            torch_dtype='auto',
            # if model_args.model_name_or_path.find("falcon") != -1 else False
            trust_remote_code=True

        )
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            device_map='auto',
            # torch_dtype='auto',
            load_in_8bit=True,
            # if model_args.model_name_or_path.find("falcon") != -1 else False
            trust_remote_code=True

        )

    if training_args.use_lora:

        logging.warning("Loading model to Lora")

        from peft import LoraConfig, get_peft_model
        LORA_R = 32
        # LORA_ALPHA = 16
        LORA_DROPOUT = 0.05
        TARGET_MODULES = [
            "o_proj","gate_proj", "down_proj", "up_proj"
        ]

        config = LoraConfig(
            r=LORA_R,
            # lora_alpha=LORA_ALPHA,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # model = model.to(torch.bfloat16)
        model = get_peft_model(model, config)
        # peft_module_casting_to_bf16(model)
        model.print_trainable_parameters()

    # model.is_parallelizable = True
    # model.model_parallel = True
    # torch.cuda.empty_cache()   
    return model


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    model = build_model(model_args, training_args, data_args)

    # with training_args.main_process_first(desc="loading and tokenization"):

    #     train_dataset = make_train_dataset(
    #         tokenizer=tokenizer, data_path=data_args.data_path, data_args=data_args)

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model,
                                           label_pad_token_id=IGNORE_INDEX
                                           )

    trainer = Trainer(model=model,
                      tokenizer=tokenizer,
                      args=training_args,
                      train_dataset=train_dataset,
                      eval_dataset=None,
                      data_collator=data_collator)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    train()
