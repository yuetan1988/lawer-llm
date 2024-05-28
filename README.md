<h1 align="center">
<img src="./docs/assets/logo.svg" width="590" align=center/>
</h1><br>


[书生·浦语 InternLM](https://github.com/InternLM) 开源大模型实践项目: 罪恶克星-法律大模型


## 项目目标

[详细文档](https://github.com/yuetan1988/lawer-llm/wiki)

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

### 激活环境
```shell
conda activate xtuner
pip install -r requirements.txt
```

### SFT

**Performance**

| Model | Size | LawBench<sup>0shotval</sup> | LawBench<sup>1shotval</sup> | weights |
| :-- | :-: | :-: | :-: | :-: | 
| internlm2-chat-7b-sft | 7B |  |  | [官方weights](https://huggingface.co/internlm/internlm2-chat-7b-sft) |
| internlm2-chat-7b | 7B |  |  |  |
| internlm2-chat-7b-4bits | 7B |  |  |  |
| internlm2-chat-20b-sft | 20B |  |  |  |
| internlm2-chat-20b | 20B |  |  |  |
| Meta-Llama-3-8B-Instruct | 8B |  |  |  |


### RAG





### web-demo

```shell
# 端口查看开发机ssh连接的端口
streamlit run web_demo.py --server.address 127.0.0.1 --server.port 38649
```
