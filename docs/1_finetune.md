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

LORA微调
```shell
cd app/finetune
sh run.sh
```