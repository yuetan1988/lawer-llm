from functools import partial

from datasets import load_dataset
import transformers
from modelscope import AutoModel, AutoTokenizer, snapshot_download
from modelscope.msdatasets import MsDataset


DATA_PATH = "../../inputs/train_data.json"


def download():
    ds = MsDataset.load(
        "AI-ModelScope/DISC-Law-SFT",
        subset_name="default",
        split="train",
        cache_dir="../../inputs",
    )

    train_dataset = ds.to_hf_dataset()
    # train_dataset = train_dataset.select(range(100))
    train_dataset.to_json(DATA_PATH)


def test_data():
    datasets = load_dataset(
        "json",
        data_files={"train": DATA_PATH},
    )
    train_dataset = datasets["train"]

    for i in range(10):
        print(train_dataset[i])


if __name__ == "__main__":
    # download()
    test_data()
