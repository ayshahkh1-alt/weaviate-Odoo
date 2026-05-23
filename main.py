from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

import os
from dotenv import load_dotenv

import weaviate
from weaviate.classes.init import Auth
from openai import OpenAI
import json

load_dotenv()

app = FastAPI()

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# WEAVIATE
# =========================
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
)

collection = client.collections.get("products")

# =========================
# OPENAI
# =========================
ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# MEMORY
# =========================
chat_memory = {}

class Question(BaseModel):
    session_id: str
    question: str


def build_memory(session_id, msg):
    if session_id not in chat_memory:
        chat_memory[session_id] = []

    chat_memory[session_id].append({"role": "user", "content": msg})

    return chat_memory[session_id][-10:]


# =========================
# CHAT ENDPOINT
# =========================
@app.post("/chat")
def chat(data: Question):

    history = build_memory(data.session_id, data.question)

    # =========================
    # INTENT
    # =========================
    intent_response = ai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
استخرج JSON فقط:
{
  "occasion": null,
  "color": null,
  "product_type": null
}
"""
            },
            {
                "role": "user",
                "content": str(history)
            }
        ]
    )

    try:
        intent = json.loads(intent_response.choices[0].message.content)
    except:
        intent = {"occasion": None, "color": None, "product_type": None}

    # =========================
    # QUERY
    # =========================
    query = data.question

    if intent.get("occasion"):
        query += " " + intent["occasion"]

    if intent.get("color"):
        query += " " + intent["color"]

    if intent.get("product_type"):
        query += " " + intent["product_type"]

    # =========================
    # SEARCH
    # =========================
    results = collection.query.near_text(
        query=query,
        limit=8,
        return_properties=[
            "name",
            "color",
            "price",
            "available",
            "image_url",
            "product_url"
        ]
    )

    if not results.objects:
        return {
            "answer": "ما لقيت نتائج 😢 جرب وصف أوضح",
            "products": []
        }

    products = []

    context = ""

    for obj in results.objects:
        p = obj.properties

        products.append(p)

        context += f"""
اسم: {p.get('name')}
لون: {p.get('color')}
سعر: {p.get('price')}
"""

    # =========================
    # FINAL RESPONSE
    # =========================
    response = ai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[

            {
                "role": "system",
                "content": """
أنت مساعد متجر زهور.

- اعرض المنتجات بشكل واضح
- لا تخترع بيانات
- استخدم فقط المعلومات المعطاة
- إذا يوجد منتجات، اعرضها بشكل مرتب
"""
            },

            {
                "role": "user",
                "content": f"""
المستخدم: {data.question}

المنتجات:
{context}
"""
            }
        ]
    )

    answer = response.choices[0].message.content

    chat_memory[data.session_id].append({
        "role": "assistant",
        "content": answer
    })

    return {
        "answer": answer,
        "products": products
    }