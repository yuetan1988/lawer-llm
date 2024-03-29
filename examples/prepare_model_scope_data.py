from modelscope.msdatasets import MsDataset
from modelscope import snapshot_download, AutoModel, AutoTokenizer


ds = MsDataset.load('AI-ModelScope/DISC-Law-SFT', subset_name='default', split='train', cache_dir='../inputs')
