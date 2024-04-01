from functools import partial

import transformers
from modelscope import AutoModel, AutoTokenizer, snapshot_download
from modelscope.msdatasets import MsDataset

ds = MsDataset.load(
    "AI-ModelScope/DISC-Law-SFT",
    subset_name="default",
    split="train",
    cache_dir="../inputs",
)

train_dataset = ds.to_hf_dataset()
# train_dataset = train_dataset.select(range(100))
train_dataset.to_json("../inputs/train_data.json")
