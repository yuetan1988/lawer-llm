from modelscope.msdatasets import MsDataset
from modelscope import snapshot_download, AutoModel, AutoTokenizer
from functools import partial
import transformers


ds = MsDataset.load('AI-ModelScope/DISC-Law-SFT', subset_name='default', split='train', cache_dir='../inputs')

train_dataset = ds.to_hf_dataset()
train_dataset = train_dataset.select(range(100))

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
 
    input_data = examples['input']
    output = examples['output']

    len_ = len(input_data)


    sources = [prompt_input.format_map({'instruction': ins_data[i], 'input': input_data[i]}) if input_data[
        i] != "" else prompt_no_input.format_map(
        {'instruction': ins_data[i]})
        for i in range(len_)]
    sources = [i[:data_args.source_length] for i in sources]
    targets = [
        f"{example[:data_args.target_length-1]}{tokenizer.eos_token}" for example in output]

    input_output = preprocess(
        sources=sources, targets=targets, tokenizer=tokenizer)
    examples['input_ids'] = input_output['input_ids']
    examples['labels'] = input_output['labels']
    return examples


train_dataset = train_dataset.map(
    function=partial(generate_sources_targets, tokenizer=tokenizer),
    batched=True,
    desc="Running tokenizer on train dataset",
    num_proc=20
).shuffle()
