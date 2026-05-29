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
# 🧠 CHAT MEMORY
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
    # ❌ EMPTY MESSAGE
    # =========================

    if len(data.question.strip()) < 2:

        return {
            "answer": "ممكن توضّح أكثر 😊",
            "products": []
        }

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

        # تجاهل النتائج الضعيفة
        if distance > 0.35:
            continue

        # =========================
        # 🌸 PRODUCT
        # =========================

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

        # =========================
        # 📚 ARTICLE
        # =========================

        elif props.get("type") == "article":

            context += f"""

[مقال]

العنوان: {props.get("title")}
المحتوى: {props.get("content")}

"""

    # =========================
    # 🤖 SYSTEM PROMPT
    # =========================

    messages = [

        {
            "role": "system",
            "content": """
أنت مساعد ذكي ومتخصص لمتجر زهور وهدايا.

تصرف كبائع محترف داخل متجر حقيقي.

تعليمات مهمة جداً:

- اقرأ المحادثة كاملة قبل الرد
- لا تكرر كلمة "تفضل"
- لا تكرر نفس الجمل
- إذا لم تفهم طلب المستخدم اسأله سؤالاً توضيحياً
- إذا لم تجد نتائج مناسبة اطلب تفاصيل أكثر
- كن ودوداً وطبيعياً
- تصرف كإنسان حقيقي

أمثلة:

إذا قال المستخدم:
"بدي بوكيه"

قل:
"أكيد 😊 لأي مناسبة بدك البوكيه؟"

إذا قال:
"بدي هدية"

قل:
"لمين الهدية؟ وكم الميزانية تقريباً؟"

إذا قال:
"بدي بوكيه خطبة"

قل:
"بتحب يكون البوكيه ناعم ولا فخم؟"

قواعد الرد:

- ممنوع HTML
- ممنوع Markdown

إذا أردت عرض صورة استخدم فقط:
IMAGE:رابط_الصورة

إذا وجدت منتجات مناسبة:
اعرضها بشكل مرتب مع وصف بسيط.
"""
        }

    ]

    # =========================
    # 🧠 ADD MEMORY
    # =========================

    messages.extend(history[-12:])

    # =========================
    # 📦 ADD CONTEXT
    # =========================

    messages.append({
        "role": "system",
        "content": f"""
معلومات من قاعدة البيانات:

{context}
"""
    })

    # =========================
    # 👤 USER MESSAGE
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
        temperature=1
    )

    answer = response.choices[0].message.content.strip()

    # =========================
    # 🚫 ANTI REPETITION
    # =========================

    bad_answers = [
        "تفضل",
        "تفضل 🌸",
        "أكيد",
        "نعم"
    ]

    if answer in bad_answers:

        answer = (
            "أكيد 😊 "
            "ممكن توضّح أكثر شو النوع أو المناسبة اللي بدك إياها؟"
        )

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
