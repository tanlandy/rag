from demo.db_class import MyVectorDBConnector
from demo.llm_class import MyLLMChain
from demo.chatbot_class import MyRAGBot

vector_db = MyVectorDBConnector(
    collection_name="hello_rag_v2", embedding_model_name="MiniLM-L6-v2"
)
llm_api = MyLLMChain(use_rag=False)
bot = MyRAGBot(vector_db, llm_api)

user_input = "虚拟语气应该如何在从句中使用？"
output = bot.chat(user_input)

print(output)
