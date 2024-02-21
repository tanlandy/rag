from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


class MyLLMChain:
    def __init__(self, use_rag=True):
        self.use_rag = use_rag
        base_url = "http://192.168.100.252:8000/v1/"
        llm = ChatOpenAI(temperature=0, api_key="EMPTY", base_url=base_url)
        self.llm_chain = self.generate_llm_chain(llm)

    def generate_llm_chain(self, llm):
        if self.use_rag:
            template = PromptTemplate(
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
        else:
            template = PromptTemplate(
                input_variables=["QUERY"],
                template="""
你是一个问答机器人。
你的任务是回答用户问题。

用户问：
{QUERY}

请用中文回答用户问题。
""",
            )

        return LLMChain(llm=llm, prompt=template)

    def response(self, query, rag_info):
        if self.use_rag:
            output = self.llm_chain.invoke({"QUERY": query, "INFO": rag_info})
        else:
            output = self.llm_chain.invoke({"QUERY": query})
        return output
