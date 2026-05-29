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

# =========================
# ✅ CORS
# =========================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# 🔗 WEAVIATE CONNECTION
# =========================

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=Auth.api_key(
        os.getenv("WEAVIATE_API_KEY")
    ),
    headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
    }
)

collection = client.collections.get("KnowledgeBase")

# =========================
# 🤖 OPENAI CLIENT
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
    product_url: str


class Article(BaseModel):
    title: str
    content: str


class Question(BaseModel):
    question: str


# =========================
# 🌸 SAVE PRODUCT
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
            "product_url": flower.product_url,
            "text": f"""
اسم المنتج: {flower.name}
اللون: {flower.color}
السعر: {flower.price}
التوفر: {flower.available}
رابط المنتج: {flower.product_url}
"""
        }
    )

    return {"status": "product saved ✔"}


# =========================
# 📚 SAVE ARTICLE
# =========================

@app.post("/update-article")
def update_article(article: Article):

    collection.data.insert(
        properties={
            "type": "article",
            "title": article.title,
            "content": article.content,
            "text": f"""
عنوان المقال: {article.title}
المحتوى: {article.content}
"""
        }
    )

    return {"status": "article saved ✔"}


# =========================
# 🤖 CHAT ENDPOINT
# =========================

@app.post("/chat")
def chat(data: Question):

    results = collection.query.near_text(
        query=data.question,
        limit=8,
        certainty=0.7
    )

    seen = set()
    filtered_objects = []

    for obj in results.objects:
        name = obj.properties.get("name") or obj.properties.get("title")

        if name in seen:
            continue

        seen.add(name)
        filtered_objects.append(obj)

    context = "=== KNOWLEDGE BASE ===\n\n"

    for obj in filtered_objects:

        props = obj.properties

        if props.get("type") == "product":
            context += f"""
------------------
TYPE: product
NAME: {props.get("name")}
COLOR: {props.get("color")}
PRICE: {props.get("price")}
AVAILABLE: {props.get("available")}
URL: {props.get("product_url")}
IMAGE: {props.get("image_url")}
"""

        elif props.get("type") == "article":
            context += f"""
------------------
TYPE: article
TITLE: {props.get("title")}
CONTENT: {props.get("content")}
"""

    response = ai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.9,
        presence_penalty=0.6,
        frequency_penalty=0.6,
        messages=[
            {
                "role": "system",
                "content": """
أنت مساعد ذكي لمتجر زهور وهدايا.

- لا تستخدم HTML
- لا تستخدم Markdown
- كن متنوع في الردود
- لا تكرر نفس الجمل
- استخدم المعلومات فقط من السياق
"""
            },
            {
                "role": "user",
                "content": f"""
السؤال:
{data.question}

البيانات:
{context}
"""
            }
        ]
    )

    return {
        "answer": response.choices[0].message.content,
        "products": [
            {
                "name": obj.properties.get("name"),
                "color": obj.properties.get("color"),
                "price": obj.properties.get("price"),
                "available": obj.properties.get("available"),
                "image_url": obj.properties.get("image_url"),
                "product_url": obj.properties.get("product_url")
            }
            for obj in filtered_objects
            if obj.properties.get("type") == "product"
        ]
    }

# =========================
# 🔚 CLOSE CONNECTION
# =========================

@app.on_event("shutdown")
def shutdown_event():
    client.close()