import os
from dotenv import load_dotenv

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType

load_dotenv()

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
)

print("Connected ✔")

# حذف قديم
if client.collections.exists("products"):
    client.collections.delete("products")
    print("Old collection deleted ✔")

# إنشاء جديد
client.collections.create(
    name="products",

    vector_config=Configure.Vectors.text2vec_openai(
        model="text-embedding-3-small"
    ),

    properties=[
        Property(name="name", data_type=DataType.TEXT),
        Property(name="color", data_type=DataType.TEXT),
        Property(name="price", data_type=DataType.NUMBER),
        Property(name="available", data_type=DataType.NUMBER),
        Property(name="image_url", data_type=DataType.TEXT),
        Property(name="product_url", data_type=DataType.TEXT),
        Property(name="text", data_type=DataType.TEXT),
    ]
)

print("Collection created ✔")

client.close()