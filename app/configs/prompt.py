class PromptCN:
    """指令"""

    # rag
    rag_prompt = "作为一个精通法律的专家, 请仔细阅读参考材料并使用中文回答问题。记住，如果你不清楚法案直接说不了解，不要试图捏造答案；我们深呼吸一下，一步一步的对问题和资料进行思考。\n 材料：“{}”\n 问题：“{}” "

    """
    COT: Lets's think step by step
    """

    KEYWORD_PROMPT = (
        "从这句话中抽取5个和法律、条例、规定相关的关键字 "
        "### \n{instruction}\n"
        "只输出关键字即可, 不要说多余的话"
    )

    TOPIC_PROMPT = "告诉我这句话的主题，直接说主题不要解释" "\n{instruction}\n"

    SCORING_QUESTION_PROMPT = (
        "{instruction}\n"
        "请仔细阅读以上内容，判断句子是否是个有主题的疑问句，结果用 0～10 表示。直接提供得分不要解释。\n判断标准：有主语谓语宾语并且是疑问句得 10 分；缺少主谓宾扣分；陈述句直接得 0 分；不是疑问句直接得 0 分。直接提供得分不要解释。"
    )

    SCORING_RELAVANCE_PROMPT = (
        "问题：{query}\n材料：{passage}\n"
        "请仔细阅读以上内容，判断问题和材料的关联度，用0～10表示。判断标准：非常相关得 10 分；完全没关联得 0 分。直接提供得分不要解释。"
    )
    SUMMARIZE_PROMPT = "{query} \n" "仔细阅读以上内容，输出其摘要，总结得简短有力"

    MULTI_QUERY_PROMPT = (
        "从这个用户问题中生成3个不同版本，以从矢量数据库中检索相关文档。通过对用户问题生成多个视角，您的目标是帮助用户克服基于距离的相似性搜索的一些限制。提供这些用换行符分隔的替代问题，不要给出多余的回答。\n"
        "问题：{query}"
    )

    HYPO_QUESTION_PROMPT = (
        "生成5个假设问题的列表，以下文档可用于回答这些问题:\n\n{passage}"
    )

    LAW_PROMPT = "你是一个专业律师，请判断下面问题是否和法律相关，相关请回答YES，不想关请回答NO，不允许其它回答，不允许在答案中添加编造成分。"

    RAG_PROMPT = """请使用提供的上下文来回答问题，总是用中文回答。如果无法从上下文中得到答案，请回答不知道。
        提供的上下文：
        ···
        {context}
        ···
        用户的问题: {question}
        你的回答:"""

    RETRIEVAL_PROMPT = "为这个句子生成表示以用于检索相关文章：{context}"