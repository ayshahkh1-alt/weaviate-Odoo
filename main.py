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
# 🔗 Weaviate Connection
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

    return {
        "status": "product saved ✔"
    }

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

    return {
        "status": "article saved ✔"
    }

# =========================
# 🔎 CHAT (RAG)
# =========================
# =========================
# 🔎 CHAT (RAG)
# =========================

@app.post("/chat")
def chat(data: Question):

    # =========================
    # 🔍 SEARCH IN WEAVIATE
    # =========================

    results = collection.query.near_text(
        query=data.question,
        limit=20
    )

    product_context = ""
    article_context = ""

    # =========================
    # 📦 BUILD CONTEXT
    # =========================

    for obj in results.objects:

        props = obj.properties

        # =====================
        # 🌸 PRODUCTS
        # =====================

        if props.get("type") == "product":

            name = props.get("name") or ""
            color = props.get("color") or "غير متوفر"
            price = props.get("price") or 0
            available = props.get("available") or 0
            image_url = props.get("image_url") or ""

            # ✅ فلترة البوكيهات فقط
            if only_bouquets and "بوكيه" not in name:
                continue

            product_context += f"""

اسم المنتج: {name}
اللون: {color}
السعر: {price}
الكمية المتوفرة: {available}
رابط الصورة: {image_url}

"""

        # =====================
        # 📚 ARTICLES
        # =====================

        elif props.get("type") == "article":

            title = props.get("title") or ""
            content = props.get("content") or ""

            article_context += f"""

عنوان المقال:
{title}

المحتوى:
{content}

"""

    # =========================
    # 🤖 OPENAI RESPONSE
    # =========================

    response = ai_client.chat.completions.create(
        model="gpt-4o-mini",

        messages=[

            {
                "role": "system",

                "content": """
أنت مساعد ذكي ومتخصص لمتجر زهور وهدايا.

- لا تخترع أي منتج.
- لا تخترع أي سعر.
- لا تخترع أي وصف.
- استخدم المنتجات الحقيقية فقط.
- إذا طلب المستخدم بوكيه اعرض البوكيهات فقط.
- إذا طلب صورة لا تقل أنك لا تستطيع عرض الصور.
- كن مختصر وواضح.
"""
            },

            {
                "role": "user",

                "content": f"""
سؤال المستخدم:
{data.question}

المنتجات:
{product_context}

المقالات:
{article_context}
"""
            }
        ]
    )

    # =========================
    # ✅ FINAL RESPONSE
    # =========================

    return {

        "answer": response.choices[0].message.content,

        "products": [

            {
                "name": obj.properties.get("name"),
                "price": obj.properties.get("price"),
                "image_url": obj.properties.get("image_url"),
                "available": obj.properties.get("available")
            }

            for obj in results.objects

            if obj.properties.get("type") == "product"
            and (
                not only_bouquets
                or "بوكيه" in (obj.properties.get("name") or "")
            )
        ]
    }
# =========================
# 🔚 CLOSE CONNECTION
# =========================

@app.on_event("shutdown")
def shutdown_event():
    client.close()