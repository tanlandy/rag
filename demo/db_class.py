from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

from sentence_transformers import SentenceTransformer


class MyVectorDBConnector:
    def __init__(self, collection_name, embedding_model_name):
        connections.connect("default", host="localhost", port="19530")
        self.model_info = {
            "MiniLM-L6-v2": (384, "/data/remote_dev/lin/all-MiniLM-L6-v2-sentence-transformer-model"),
            "mpnet-base-v2": (768, "/data/remote_dev/lin/all-mpnet-base-v2")
        }
        self.dim, self.embedding_model_name = self.check_collection_model(collection_name, embedding_model_name)
        self.embedding_model = SentenceTransformer(self.embedding_model_name)
        if not utility.has_collection(collection_name):  # if collection not exist, create it
            fields = [
                FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=100),
                FieldSchema(name="text", dtype=DataType.VARCHAR, description="this is the original text field", max_length=5000), 
                FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
            ]
            schema = CollectionSchema(fields)
            self.collection = Collection(collection_name, schema, consistency_level="Strong")
        else:
            self.collection = Collection(collection_name)
            self.collection.load()


    
    def check_collection_model(self, collection_name, embedding_model_name):
        valid_collections = {
            "MiniLM-L6-v2": ["hello_rag_v2", "hello_rag_v3"],
            "mpnet-base-v2": ["hello_rag_v4"]
        }

        # Check if the embedding model is valid
        if embedding_model_name not in self.model_info:
            raise ValueError(f"embedding model {embedding_model_name} not found")

        # Check if the collection name is valid for the embedding model
        if collection_name not in valid_collections.get(embedding_model_name, []):
            if utility.has_collection(collection_name):
                raise ValueError(f"collection {collection_name} and embedding model {embedding_model_name} are not in pairs")

        # Get the dimension and path for the embedding model
        dim, path_name = self.model_info[embedding_model_name]
        return dim, path_name


    def insert(self, chunks):
        # embed data
        embeddings = self.embedding_model.encode(chunks)

        # insert entities(data) to collection
        start = len(self.collection.num_entities)
        end = start + len(chunks)
        entities = [
            [str(i) for i in range(start, end)], # primary key
            chunks,
            embeddings,    # field embeddings, supports numpy.ndarray and list
        ]
        self.collection.insert(entities)
        self.collection.flush()
        print(f"number of entities inserted: {self.collection.num_entities}")

        # index the collection
        index = {
            "index_type": "GPU_IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128},
        }
        self.collection.create_index("embeddings", index)


    def search(self, query):
        vectors_to_search = self.embedding_model.encode([query])
        search_params = {
            "metric_type": "COSINE", 
            "offset": 0, 
            "ignore_growing": False, 
            "params": {"nlist": 128},
        }
        res = self.collection.search(vectors_to_search, "embeddings", search_params, limit=3, expr=None, output_fields=["text"], partition_names=None)
        return res[0][0].text

