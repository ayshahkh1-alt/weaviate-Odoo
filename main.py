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

    question = data.question.lower()

    only_bouquets = False

    if "بوكيه" in question:
        only_bouquets = True

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

            # ✅ إذا المستخدم طلب بوكيه
            # لا تعرض ورد مفرد
            if only_bouquets and "بوكيه" not in name:
                continue

            color = props.get("color") or "غير متوفر"
            price = props.get("price") or "غير متوفر"
            available = props.get("available") or "غير متوفر"
            image_url = props.get("image_url") or ""

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

المعلومات التي تصلك تأتي من:
- منتجات متجر حقيقية
- مقالات قاعدة المعرفة

مهامك:

1- فهم سؤال المستخدم بدقة.

2- إذا كان السؤال غير واضح:
اسأل سؤال توضيحي قبل إعطاء الإجابة.

3- إذا كان السؤال عن منتج:
اذكر:
- اسم المنتج
- السعر
- الألوان المتوفرة
- التوفر

- المنتجات هي المصدر الحقيقي الوحيد للأسعار والمخزون والتوفر.
- لا تذكر أي منتج غير موجود ضمن المنتجات الحقيقية المرسلة لك.
- ممنوع اختراع أي سعر أو كمية أو لون أو توفر.

- إذا كانت المعلومة غير موجودة اكتب:
غير متوفر

- لا تخترع وصف تسويقي.

- إذا لم تجد معلومات كافية قل ذلك بوضوح.

- كن مختصر وواضح.

- إذا وُجدت عدة منتجات:
رتبها بشكل مرتب وسهل القراءة.

4- إذا كان السؤال يعتمد على مقالات قاعدة المعرفة:
لا تنسخ النص حرفيًا.
استخرج المعلومة المهمة ثم اشرحها بأسلوب طبيعي ومختصر.

5- إذا كان السؤال عن مناسبة:
اقترح منتجات مناسبة حسب المناسبة.

6- إذا طلب المستخدم صورة:
لا تقل أنك لا تستطيع عرض الصور.
الصور يتم عرضها تلقائيًا من النظام.

7- إذا كان المستخدم يريد بوكيه:
لا تقترح ورد مفرد.
"""
            },

            {
                "role": "user",

                "content": f"""
سؤال المستخدم:
{data.question}

المنتجات الحقيقية:
=========================

{product_context}

=========================

مقالات المعرفة:
=========================

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
        ]
    }
# =========================
# 🔚 CLOSE CONNECTION
# =========================

@app.on_event("shutdown")
def shutdown_event():
    client.close()