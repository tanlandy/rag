from pymilvus import (
    connections,
    Collection,
)

# 1. load vectorDB collection
connections.connect("default", host="localhost", port="19530")
hello_rag_v2 = Collection("hello_rag_v2")
hello_rag_v2.load()

# |   | field name   | field type  | other attributes              | field description          |
# |---|--------------|-------------|-------------------------------|----------------------------|
# | 1 | "pk"         | VarChar     | is_primary=True auto_id=False | "primary field"            |
# | 2 | "text"       | VarChar     |                               | "original text"            |
# | 3 | "embeddings" | FloatVector | dim=384                       | "float vector with dim 384"|

# 2. load sentence transformer model

from sentence_transformers import SentenceTransformer

embedding_model_name = "/data/remote_dev/lin/all-MiniLM-L6-v2-sentence-transformer-model"  # dim = 384, loaded from local on the server GPU1
embedding_model = SentenceTransformer(embedding_model_name)

# 3. get text and embeddings

search_params = {
    "metric_type": "COSINE",
    "offset": 0,
    "ignore_growing": False,
    "params": {"nlist": 128},
}


def search(user_query: str, topk=3) -> str:
    print(f"user query: {user_query}")
    vectors_to_search = embedding_model.encode([user_query])
    res = hello_rag_v2.search(
        vectors_to_search,
        "embeddings",
        search_params,
        limit=topk,
        expr=None,
        output_fields=["text"],
        partition_names=None,
    )
    print(f"grammar book search result: \n{'-' * 50}\n{res[0][0].text}\n{'-' * 50}")
    return res[0][0].text


# 4. create LLMChain

from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

base_url = "http://192.168.100.252:8000/v1/"
llm = ChatOpenAI(temperature=0, api_key="EMPTY", base_url=base_url)

rag_template = PromptTemplate(
    input_variables=["INFO", "QUERY"],
    template="""
你是一个问答机器人。
你的任务是根据下述给定的已知信息回答用户问题。
确保你的回复完全依据下述已知信息。不要编造答案。
如果下述已知信息不足以回答用户的问题，请直接回复"我无法回答您的问题"。

已知信息:
{INFO}

用户问：
{QUERY}

请用中文回答用户问题。
""",
)

rag_chain = LLMChain(llm=llm, prompt=rag_template)

user_input = "虚拟语气应该如何在从句中使用？"
# user_input = "今天天气怎么样？"  # 我无法回答您的问题。
# user_input = "使动用法是什么?"
# user_input = "英语中虚拟语气是什么？"
# user_input = "英语中定语从句是什么？"
# user_input = "过去完成进行时是什么？"
# user_input = "请给我10个不规则动词的过去式的例子"
rag_info = search(user_input)
output = rag_chain.invoke({"QUERY": user_input, "INFO": rag_info})

print(output["text"])
