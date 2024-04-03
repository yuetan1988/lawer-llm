from datasets import load_dataset
from modelscope import MsDataset
from modelscope.utils.constant import DownloadMode


try:
    dataset = load_dataset("Anthropic/hh-rlhf")
    dataset["train"].to_json("../../inputs/hh-rlhf2/train_hhrlhf.json")
    dataset["test"].to_json("../../inputs/hh-rlhf2/test_hhrlhf.json")
except:
    ds = MsDataset.load(
        "AI-ModelScope/hh_rlhf_cn",
        subset_name="harmless_base_cn",
        download_mode=DownloadMode.FORCE_REDOWNLOAD,
    )
    ds_train = ds["train"].to_hf_dataset()
    ds_train.to_json("../../inputs/hh-rlhf2/train_hhrlhf.json")
    print(next(iter(ds["train"])))

    ds_test = ds["test"].to_hf_dataset()
    ds_test.to_json("../../inputs/hh-rlhf2/test_hhrlhf.json")
