from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
from dotenv import load_dotenv

import weaviate
from weaviate.classes.init import Auth

from openai import OpenAI

load_dotenv()

app = FastAPI()

# ✅ CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 🔗 Weaviate Connection
# =========================

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
    headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
    }
)

collection = client.collections.get("KnowledgeBase")

# =========================
# 🤖 OpenAI Client
# =========================

ai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# =========================
# 🟢 MODELS
# =========================

class Flower(BaseModel):
    name: str
    color: str
    price: float
    available: float
    image_url: str


class Article(BaseModel):
    title: str
    content: str


class Question(BaseModel):
    question: str


# =========================
# 🟣 PRODUCTS
# =========================

@app.post("/update-flower")
def update_flower(flower: Flower):

    collection.data.insert(
        properties={
            "type": "product",
            "name": flower.name,
            "color": flower.color,
            "price": flower.price,
            "available": flower.available,
            "image_url": flower.image_url,
            "text": f"""
اسم المنتج: {flower.name}
اللون: {flower.color}
السعر: {flower.price}
الكمية المتوفرة: {flower.available}
"""
        }
    )

    return {"status": "product saved ✔"}


# =========================
# 🟣 ARTICLES
# =========================

@app.post("/update-article")
def update_article(article: Article):

    collection.data.insert(
        properties={
            "type": "article",
            "title": article.title,
            "content": article.content,
            "text": f"""
عنوان المقال:
{article.title}

المحتوى:
{article.content}
"""
        }
    )

    return {"status": "article saved ✔"}


# =========================
# 🔎 CHAT (RAG - ANSWER ONLY)
# =========================

@app.post("/chat")
def chat(data: Question):

    results = collection.query.near_text(
        query=data.question,
        limit=8
    )

    context = ""

    for obj in results.objects:

        props = obj.properties

        if props.get("type") == "product":

            context += f"""
[منتج]
الاسم: {props.get("name")}
اللون: {props.get("color")}
السعر: {props.get("price")}
التوفر: {props.get("available")}
"""

        elif props.get("type") == "article":

            context += f"""
[مقال]
العنوان: {props.get("title")}
المحتوى: {props.get("content")}
"""

    response = ai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[

            {
                "role": "system",
                "content": """
أنت مساعد ذكي لمتجر زهور وهدايا.

استخدم المعلومات المرفقة فقط.
إذا لم تجد معلومة كافية، قل ذلك بصراحة.
لا تخترع معلومات.
كن مختصر وواضح ومفيد.
"""
            },

            {
                "role": "user",
                "content": f"""
سؤال المستخدم:
{data.question}

المعلومات:
{context}
"""
            }
        ]
    )

    # =========================
    # ✅ RETURN ONLY ANSWER
    # =========================

    return {
        "answer": response.choices[0].message.content
    }


# =========================
# 🔚 CLOSE CONNECTION
# =========================

@app.on_event("shutdown")
def shutdown_event():
    client.close()