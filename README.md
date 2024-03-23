# LLM律师小助手

书生·浦语 (InternLM) 开源大模型实践项目。

## 项目目标

**Vision**: 大语言模型助力全面依法治国

**Objective**: 
- 短期目标: 根据已有资料，回答法律相关问题
- 长期目标: 能够根据上传资料，结合法律条文，给出缺失项与违法项


## 技术路线

- continue pretrain
    - mixed data, hybrid-tuning
- fine-tune
    - 数据: 配比, diversity, Self-Instruct, Self-QA, Self-KG
    - 减少幻觉: Generate with Citation, Factual Consistency Evaluation
- RAG
- Agent
- TBD


## How to use it
同步模型权重
```
```

激活环境
```
conda activate xtuner0.1.9
```

web-demo
```
# 端口查看开发机ssh连接的端口
streamlit run web_demo.py --server.address 127.0.0.1 --server.port 38649
```

sft
```
python llm_finetune.py
```



## 资料

- https://github.com/PKU-YuanGroup/ChatLaw
- https://github.com/FudanDISC/DISC-LawLLM
- https://github.com/CSHaitao/LexiLaw
- https://github.com/CSHaitao/Awesome-LegalAI-Resources
- https://github.com/lvwzhen/law-cn-ai
- https://github.com/LiuHC0428/LAW-GPT
