import copy
import logging
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional

import transformers
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
)

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
    num_train_epochs = (1,)
    per_device_train_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=4)
    optim: str = field(default="paged_adamw_32bit")  # "adamw_torch"
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


def build_sft_data(data_args):
    datasets = load_dataset(
        "json",
        data_files={"train": data_args.data_path},
        # cache_dir=cache_dir,
    )
    train_dataset = datasets["train"]
    return train_dataset


def build_prompt(examples, prompt_input, tokenizer, data_args):
    # 注意examples的结构, 包含多个keys, 也就是df的列信息. 而其中的instruction/response等长度则是行信息

    if "instruction" in examples:
        ins_data = examples["instruction"]
        input_data = examples["input"]
    else:
        ins_data = examples["input"]
        input_data = [""] * len(ins_data)
    output = examples["output"]

    len_ = len(ins_data)
    sources = [
        (
            prompt_input.format_map(
                {"instruction": ins_data[i], "input": input_data[i]}
            )
            if input_data[i] != ""
            else prompt_input.format_map({"instruction": ins_data[i]})
        )
        for i in range(len_)
    ]
    sources = [i[: data_args.source_length] for i in sources]
    targets = [
        f"{example[:data_args.target_length-1]}{tokenizer.eos_token}"
        for example in output
    ]
    return sources, targets


def tokenize_fn(strings, tokenizer, IGNORE_INDEX):
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
    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    ne_pad_token_id = (
        IGNORE_INDEX if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    )
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(ne_pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def preprocess(examples, tokenizer, data_args):
    sources, targets = build_prompt(
        examples, prompt_input=PROMPT, tokenizer=tokenizer, data_args=data_args
    )

    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [
        tokenize_fn(strings, tokenizer, IGNORE_INDEX) for strings in (examples, sources)
    ]
    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)
    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX
    return dict(input_ids=input_ids, labels=labels)


def build_model(
    model_args: ModelArguments,
    training_args: TrainingArguments,
) -> tuple:
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        torch_dtype="auto",
        # if model_args.model_name_or_path.find("falcon") != -1 else False
        trust_remote_code=True,
    )
    # print(model)

    if training_args.use_lora:
        from peft import LoraConfig, get_peft_model

        lora_alpha = 128
        LORA_DROPOUT = 0.05
        TARGET_MODULES = [
            "wqkv",
            "wo",
            "gate_up_proj",
            "w2",
        ]
        # TARGET_MODULES = [
        #     "q_proj",
        #     "k_proj",
        #     "v_proj",
        #     "o_proj",
        # ]

        config = LoraConfig(
            lora_alpha=lora_alpha,
            target_modules=TARGET_MODULES,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        # model = model.to(torch.bfloat16)
        model = get_peft_model(model, config)
        model.print_trainable_parameters()

    # model.is_parallelizable = True
    # model.model_parallel = True
    # torch.cuda.empty_cache()
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
    tokenizer.padding_side = "right"

    train_dataset = build_sft_data(data_args)

    train_dataset = train_dataset.map(
        function=partial(preprocess, tokenizer=tokenizer, data_args=data_args),
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Running tokenizer on train dataset",
        num_proc=20,
    ).shuffle()

    model = build_model(model_args, training_args)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer, model=model, label_pad_token_id=IGNORE_INDEX
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


def lora_save():
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Merge base model with the adapter
    model = PeftModel.from_pretrained(base_model, "final_checkpoint")
    model = model.merge_and_unload()

    # Save model and tokenizer
    model.save_pretrained(new_model)
    tokenizer.save_pretrained(new_model)


def batch_generate(
    text_input: List[str],
    model,
    tokenizer,
    use_train_model: bool = True,
    temp: float = 0.7,
):
    text_input_format = [
        PROMPT.format_map({"instruction": input}) for input in text_input
    ]

    batch_inputs = tokenizer.batch_encode_plus(
        text_input_format, padding="longest", return_tensors="pt"
    )
    batch_inputs["input_ids"] = batch_inputs["input_ids"].cuda()
    batch_inputs["attention_mask"] = batch_inputs["attention_mask"].cuda()

    if use_train_model:
        # with model.disable_adapter():
        outputs = model.generate(
            **batch_inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=temp,
            top_p=0.8,
        )
    else:
        with model.disable_adapter():
            outputs = model.generate(
                **batch_inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=temp,
                top_p=0.8,
            )
    outputs = tokenizer.batch_decode(
        outputs.cpu()[:, batch_inputs["input_ids"].shape[-1] :],
        skip_special_tokens=True,
    )

    return outputs


def infer():
    base_model_name_or_path = "../../models/"
    lora_model_name_or_path = "../../outputs/result/checkpoint-100"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name_or_path,
        torch_dtype="auto",
        # device_map="auto",
        trust_remote_code=True,
    ).cuda(0)

    from peft import AutoPeftModelForCausalLM, PeftModel

    model = PeftModel.from_pretrained(model, model_id=lora_model_name_or_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name_or_path, trust_remote_code=True, padding_side="left"
    )

    test_input = ["张三怒杀阎婆惜, 该当何罪"]
    outputs = batch_generate(test_input, model, tokenizer)
    print(outputs)

    # model = AutoPeftModelForCausalLM.from_pretrained(
    #     args.output_dir,
    #     torch_dtype=torch.float16,
    #     low_cpu_mem_usage=True,
    # )
    # merged_model = model.merge_and_unload()
    # merged_model.save_pretrained(args.output_dir,safe_serialization=True, max_shard_size="2GB")
    return


if __name__ == "__main__":
    train()
    # infer()
