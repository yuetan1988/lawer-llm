# 微调

## 环境
```shell
pip install -r requirements.txt
```


## 1. 准备数据

[**DISC-LawLLM**](https://github.com/FudanDISC/DISC-LawLLM)


国服下载
```shell
cd app/finetune
python datasets/prepare_model_scope_data.py
```

[**CrimeKgAssitant**](https://github.com/liuhuanyong/CrimeKgAssitant)


## 2. 数据处理
扩充词表[可选]


## 3. 微调

全量微调
```shell
cd app/finetune
sh run.sh
```

## 4. 评测

- https://github.com/open-compass/LawBench
```shell
# 评测数据准备
cd inputs
git clone https://github.com/open-compass/LawBench.git

# 评测结果准备
cd app/finetune
python eval.py

# 分数计算
cd inputs/LawBench/evaluation
python main.py --input_folder /root/lawer-llm/outputs/finetune-eval  --outfile /root/lawer-llm/outputs/finetune-eval-res
```


- 使用[opencompass](https://github.com/open-compass/opencompass)进行评测


```shell
# 安装opencompass
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .

# 添加自定义数据: LawBench
git clone https://gitee.com/ljn20001229/LawBench.git
cp -r /root/personal_assistant/LawBench/data/one_shot /root/personal_assistant/opencompass/data/lawbench
cp -r /root/personal_assistant/LawBench/data/zero_shot /root/personal_assistant/opencompass/data/lawbench


# 评测
python run.py --datasets ceval_ppl mmlu_ppl \
--hf-path huggyllama/llama-7b \  # HuggingFace 模型地址
--tokenizer-path /root/personal_assistant/config/a/work_dirs/hf_merge\ 
--model-kwargs device_map='auto' \  # 构造 model 的参数
--tokenizer-kwargs padding_side='left' truncation='left' use_fast=False \  # 构造 tokenizer 的参数
--max-out-len 100 \  # 最长生成 token 数
--max-seq-len 2048 \  # 模型能接受的最大序列长度
--batch-size 8 \  # 批次大小
--no-batch-padding \  # 不打开 batch padding，通过 for loop 推理，避免精度损失
--num-gpus 1
```
