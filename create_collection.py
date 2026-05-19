import os
from dotenv import load_dotenv

import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure

load_dotenv()

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),

    auth_credentials=Auth.api_key(
        os.getenv("WEAVIATE_API_KEY")
    ),

    headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
    }
)

print("Connected ✔")


# حذف القديم
if client.collections.exists("KnowledgeBase"):
    client.collections.delete("KnowledgeBase")


# إنشاء الجديد مع OpenAI embeddings
client.collections.create(
    name="KnowledgeBase",

    vectorizer_config=Configure.Vectorizer.text2vec_openai()
)

print("Collection created ✔")

client.close()