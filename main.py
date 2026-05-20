from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
from dotenv import load_dotenv

import weaviate
from weaviate.classes.init import Auth

from openai import OpenAI

# =========================
# 🔄 LOAD ENV
# =========================

load_dotenv()

# =========================
# 🚀 FASTAPI
# =========================

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


class Article(BaseModel):
    title: str
    content: str


class Question(BaseModel):
    question: str

# =========================
# 🌸 UPDATE FLOWER
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
# 📚 UPDATE ARTICLE
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
# 🤖 CHAT (RAG)
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

    context = ""

    # =========================
    # 📦 BUILD CONTEXT
    # =========================

    for obj in results.objects:

        props = obj.properties

        # =====================
        # 🌸 PRODUCTS
        # =====================

        if props.get("type") == "product":

            context += f"""
[منتج]

الاسم:
{props.get("name")}

اللون:
{props.get("color")}

السعر:
{props.get("price")}

التوفر:
{props.get("available")}

رابط الصورة:
{props.get("image_url")}
"""

        # =====================
        # 📚 ARTICLES
        # =====================

        elif props.get("type") == "article":

            context += f"""
[مقال]

العنوان:
{props.get("title")}

المحتوى:
{props.get("content")}
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

استخدم المعلومات المرفقة فقط.

ممنوع اختراع:
- منتجات
- أسعار
- ألوان
- توفر

إذا طلب المستخدم صورة:
لا تقل أنك لا تستطيع عرض الصور.

إذا كانت هناك صور ضمن المنتجات:
اذكر المنتج بشكل طبيعي.

كن مختصر وواضح ولطيف.
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
    # ✅ RETURN RESPONSE
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
        ]
    }

# =========================
# 🔚 SHUTDOWN
# =========================

@app.on_event("shutdown")
def shutdown_event():
    client.close()