from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
from dotenv import load_dotenv

import weaviate
from weaviate.classes.init import Auth

from openai import OpenAI

# =========================
# 🔥 LOAD ENV
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
# 🧠 MEMORY STORE
# =========================

chat_memory = {}

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
    session_id: str
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
    # 🧠 MEMORY
    # =========================

    if data.session_id not in chat_memory:
        chat_memory[data.session_id] = []

    history = chat_memory[data.session_id]

    # =========================
    # 🔍 SEARCH
    # =========================

    results = collection.query.near_text(
        query=data.question,
        limit=8,
        return_metadata=["distance"]
    )

    # =========================
    # 📦 BUILD CONTEXT
    # =========================

    context = ""

    filtered_products = []

    for obj in results.objects:

        props = obj.properties
        distance = obj.metadata.distance

        # Ignore weak results
        if distance > 0.35:
            continue

        if props.get("type") == "product":

            filtered_products.append({
                "name": props.get("name"),
                "color": props.get("color"),
                "price": props.get("price"),
                "available": props.get("available"),
                "image_url": props.get("image_url"),
                "product_url": props.get("product_url")
            })

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
    # 🧠 BUILD MESSAGES
    # =========================

    messages = [

        {
            "role": "system",
            "content": """
أنت مساعد ذكي ومتخصص لمتجر زهور وهدايا.

تصرف كبائع محترف داخل متجر حقيقي.

قواعد مهمة جداً:

- اقرأ كامل المحادثة قبل الرد
- إذا لم تفهم طلب المستخدم اسأله سؤال توضيحي
- لا تخترع منتجات غير موجودة
- إذا لم تجد نتائج مناسبة اطلب توضيحاً
- كن ودوداً ومختصراً
- ساعد المستخدم على اختيار أفضل هدية
- اقترح منتجات مناسبة حسب المناسبة والميزانية
- حاول البيع بطريقة لطيفة واحترافية

تنسيق الرد:

- ممنوع HTML
- ممنوع Markdown
- إذا أردت عرض صورة استخدم فقط:
IMAGE:رابط_الصورة

- يمكن وضع الصور داخل الرد
- اجعل الرد مرتباً وواضحاً
"""
        }

    ]

    # =========================
    # 🧠 ADD HISTORY
    # =========================

    messages.extend(history[-12:])

    # =========================
    # 📦 ADD CONTEXT
    # =========================

    if context:

        messages.append({
            "role": "system",
            "content": f"""
معلومات من قاعدة البيانات:

{context}
"""
        })

    # =========================
    # ❓ USER MESSAGE
    # =========================

    messages.append({
        "role": "user",
        "content": data.question
    })

    # =========================
    # 💾 SAVE USER MESSAGE
    # =========================

    history.append({
        "role": "user",
        "content": data.question
    })

    # =========================
    # 🤖 OPENAI RESPONSE
    # =========================

    response = ai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.7
    )

    answer = response.choices[0].message.content

    # =========================
    # 💾 SAVE ASSISTANT MESSAGE
    # =========================

    history.append({
        "role": "assistant",
        "content": answer
    })

    # =========================
    # ✂️ LIMIT MEMORY
    # =========================

    if len(history) > 20:
        chat_memory[data.session_id] = history[-20:]

    # =========================
    # ✅ RESPONSE
    # =========================

    return {
        "answer": answer,
        "products": filtered_products
    }


# =========================
# 🧹 CLEAR MEMORY
# =========================

@app.delete("/clear-memory/{session_id}")
def clear_memory(session_id: str):

    if session_id in chat_memory:
        del chat_memory[session_id]

    return {
        "status": "memory cleared ✔"
    }


# =========================
# 🔚 CLOSE CONNECTION
# =========================

@app.on_event("shutdown")
def shutdown_event():
    client.close()
