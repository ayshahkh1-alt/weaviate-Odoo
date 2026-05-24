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

    # =========================
    # 🔍 SEARCH
    # =========================

    results = collection.query.near_text(
        query=data.question,
        limit=8
    )

    # =========================
    # 📦 BUILD CONTEXT (NO HTML)
    # =========================

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
رابط المنتج: {props.get("product_url")}
الصورة: IMAGE:{props.get("image_url")}

"""

        elif props.get("type") == "article":

            context += f"""
[مقال]

العنوان: {props.get("title")}
المحتوى: {props.get("content")}

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
أنت مساعد ذكي لمتجر زهور وهدايا.

تعليمات مهمة:

- لا تستخدم HTML نهائياً
- لا تستخدم Markdown
- إذا أردت عرض صورة استخدم فقط:

IMAGE:رابط_الصورة

- يمكن وضع الصورة داخل النص في أي مكان
- اكتب ردود قصيرة وواضحة
- إذا يوجد منتجات متعددة اعرضها بشكل مرتب
- كن مساعد متجر حقيقي وودود
"""
            },

            {
                "role": "user",
                "content": f"""
سؤال المستخدم:
{data.question}

البيانات:
{context}
"""
            }
        ]
    )

    # =========================
    # ✅ RESPONSE
    # =========================

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
            for obj in results.objects
            if obj.properties.get("type") == "product"
        ]
    }


# =========================
# 🔚 CLOSE CONNECTION
# =========================

@app.on_event("shutdown")
def shutdown_event():
    client.close()