<h1 align="center">
<img src="./docs/assets/logo.svg" width="490" align=center/>
</h1><br>


> [书生·浦语 InternLM](https://github.com/InternLM) 开源大模型实践项目: 罪恶克星-法律大模型


[WIKI文档](https://github.com/yuetan1988/lawer-llm/wiki)


## 项目目标

**Vision**: 大语言模型助力全面依法治国

**Feature**: 
- 自由回答法律问题
- 根据上传资源，回答法律问题
- 上网搜索，挥发法律问题
- 根据上传资源，提供一定审计功能


## 技术路线

![archicheture](./docs/assets/)


- continue pretrain
    - mixed data, hybrid-tuning
    - sentence-piece 词表扩充
- fine-tune
    - 数据: 配比, diversity, Self-Instruct, Self-QA, Self-KG
    - 减少幻觉: Generate with Citation, Factual Consistency Evaluation
- RAG
    - multi-vector
    - rerank
- Agent


## How to use it

### 同步模型权重
```shell
ln -s /root/share/new_models/Shanghai_AI_Laboratory/internlm2-chat-7b /root/lawer-llm/models

# cp -r /root/share/temp/model_repos/internlm-chat-7b /root/lawer-llm/models
```

### 激活环境
```shell
conda activate xtuner0.1.9
```

### web-demo
```shell
# 端口查看开发机ssh连接的端口
streamlit run web_demo.py --server.address 127.0.0.1 --server.port 38649
```

### sft

数据
```shell
git clone https://huggingface.co/datasets/ShengbinYue/DISC-Law-SFT
git clone https://github.com/liuhuanyong/CrimeKgAssitant
```

```shell
python llm_finetune.py
```

评测
- https://github.com/open-compass/LawBench


## 资料

- https://github.com/PKU-YuanGroup/ChatLaw
- https://github.com/FudanDISC/DISC-LawLLM
- https://github.com/CSHaitao/LexiLaw
- https://github.com/CSHaitao/Awesome-LegalAI-Resources
- https://github.com/lvwzhen/law-cn-ai
- https://github.com/LiuHC0428/LAW-GPT
- https://github.com/zhihaiLLM/wisdomInterrogatory
- https://github.com/billvsme/law_ai
