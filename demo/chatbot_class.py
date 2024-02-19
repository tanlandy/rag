class MyRAGBot:
    def __init__(self, vector_db, llm_api):
        self.vector_db = vector_db
        self.llm_api = llm_api

    def chat(self, user_query):
        # search for the most similar text in the database
        rag_info = self.vector_db.search(user_query)

        # generate response
        output = self.llm_api.response(user_query, rag_info)
        return output["text"]
