import os
from dotenv import load_dotenv

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType

# =========================
# 🔐 Load env
# =========================
load_dotenv()

WEAVIATE_URL = os.getenv("WEAVIATE_URL")
WEAVIATE_API_KEY = os.getenv("WEAVIATE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# =========================
# 🔗 Connect
# =========================
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=WEAVIATE_URL,
    auth_credentials=Auth.api_key(WEAVIATE_API_KEY),
    headers={
        "X-OpenAI-Api-Key": OPENAI_API_KEY
    }
)

print("Connected ✔")

# =========================
# 🧹 Delete old collection
# =========================
if client.collections.exists("products"):
    client.collections.delete("products")
    print("Old collection deleted ✔")

# =========================
# 📦 Create collection (FIXED)
# =========================
client.collections.create(
    name="products",

    vector_config=Configure.Vectors.text2vec_openai(
        model="text-embedding-3-small"
    ),

    properties=[
        Property(name="title", data_type=DataType.TEXT),
        Property(name="description", data_type=DataType.TEXT),
        Property(name="price", data_type=DataType.NUMBER),
        Property(name="image_url", data_type=DataType.TEXT),
    ]
)

print("Collection created ✔")

# =========================
# 🔚 Close safely
# =========================
client.close()
print("Connection closed ✔")