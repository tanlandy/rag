from pymilvus import (
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)

from tutorials.helpers import extract_text_from_pdf, split_text

connections.connect("default", host="localhost", port="19530")

# 2. create collection

# |   | field name   | field type  | other attributes              | field description         |
# |---|--------------|-------------|-------------------------------|---------------------------|
# | 1 | "pk"         | VarChar     | is_primary=True auto_id=False | "primary field"           |
# | 2 | "text"       | VarChar     |                               | "original text"           |
# | 3 | "embeddings" | FloatVector | dim=384                       | "float vector with dim 8" |


dim = 384
fields = [
    FieldSchema(
        name="pk",
        dtype=DataType.VARCHAR,
        is_primary=True,
        auto_id=False,
        max_length=100,
    ),
    FieldSchema(
        name="text",
        dtype=DataType.VARCHAR,
        description="this is the original text field",
        max_length=5000,
    ),
    FieldSchema(name="embeddings", dtype=DataType.FLOAT_VECTOR, dim=dim),
]

schema = CollectionSchema(
    fields, description="hello_rag_v3 is used to search similar text in a grammar book."
)
hello_rag_v3 = Collection("hello_rag_v3", schema, consistency_level="Strong")


# 3. insert data

## 3.1 prepare data
print(f"start inserting data to collection {hello_rag_v3.name}...")
import os

folder_path = "/data/remote_dev/lin/rag/grammar_book"
pdf_files = [file for file in os.listdir(folder_path) if file.endswith(".pdf")]
total_chunks = []

for pdf_file in pdf_files:
    print(f"processing {pdf_file}...")
    file_path = os.path.join(folder_path, pdf_file)
    paragraphs = extract_text_from_pdf(file_path, min_line_length=10)
    total_chunks = split_text(paragraphs, total_chunks, 300, 100)


## 3.2 embed data
print(f"start embedding data...")
from sentence_transformers import SentenceTransformer

embedding_model_name = "/data/remote_dev/lin/all-MiniLM-L6-v2-sentence-transformer-model"  # dim = 384, loaded from local on the server GPU1
embedding_model = SentenceTransformer(embedding_model_name)

embeddings = embedding_model.encode(total_chunks)

## 3.3 insert entities(data) to collection
print(f"start inserting entities to collection {hello_rag_v3.name}...")
entities = [
    [
        str(i) for i in range(len(embeddings))
    ],  # provide the pk field because `auto_id` is set to False
    total_chunks,
    embeddings,  # field embeddings, supports numpy.ndarray and list
]

insert_res = hello_rag_v3.insert(entities)
hello_rag_v3.flush()
print(f"number of entities inserted: {hello_rag_v3.num_entities}")


## 3.4 index data
print(f"start indexing data...")
index = {
    "index_type": "GPU_IVF_FLAT",
    "metric_type": "COSINE",
    "params": {"nlist": 128},
}

hello_rag_v3.create_index("embeddings", index)
print(f"index created: {hello_rag_v3.has_index}")
