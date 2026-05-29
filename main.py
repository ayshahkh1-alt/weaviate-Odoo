
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
# 🧠 MEMORY
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

    return {
        "status": "product saved ✔"
    }


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

    return {
        "status": "article saved ✔"
    }


# =========================
# 🤖 CHAT
# =========================

@app.post("/chat")
def chat(data: Question):

    question = data.question.strip()

    # =========================
    # ❌ EMPTY MESSAGE
    # =========================

    if len(question) < 2:

        return {
            "answer": "ممكن توضّح أكثر 😊",
            "products": []
        }

    # =========================
    # 🧠 MEMORY INIT
    # =========================

    if data.session_id not in chat_memory:
        chat_memory[data.session_id] = []

    history = chat_memory[data.session_id]

    # =========================
    # 🔍 SEARCH
    # =========================

    results = collection.query.near_text(
        query=question,
        limit=8,
        return_metadata=["distance"]
    )

    # =========================
    # 📦 BUILD CONTEXT
    # =========================

    context = ""

    filtered_products = []

    allowed_product_names = []

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

            allowed_product_names.append(
                props.get("name")
            )

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
    # ❌ NO PRODUCTS
    # =========================

    if len(filtered_products) == 0:

        return {
            "answer": (
                "ممكن توضّح أكثر 😊 "
                "شو نوع الهدية أو المناسبة اللي بدك إياها؟"
            ),
            "products": []
        }

    # =========================
    # ✅ ALLOWED PRODUCTS
    # =========================

    allowed_products_text = ", ".join(
        allowed_product_names
    )

    # =========================
    # 🤖 SYSTEM PROMPT
    # =========================

    messages = [

        {
            "role": "system",
            "content": f"""
أنت مساعد ذكي ومتخصص لمتجر زهور وهدايا.

تصرف كبائع محترف داخل متجر حقيقي.

تعليمات مهمة جداً:

- اقرأ المحادثة كاملة قبل الرد
- لا تكرر نفس الجمل
- لا تكرر نفس الاقتراحات
- كن طبيعي جداً
- إذا لم تفهم المستخدم اسأله سؤال توضيحي
- لا ترد بنفس الصيغة كل مرة

مهم جداً:

- ممنوع اقتراح أي منتج غير موجود
- ممنوع اختراع منتجات
- اعرض فقط المنتجات الموجودة داخل البيانات

المنتجات المسموح ذكرها فقط:

{allowed_products_text}

إذا لم تجد منتج مناسب:
اطلب تفاصيل أكثر.

إذا أراد المستخدم مناسبة:
اسأله عن:
- المناسبة
- اللون
- الميزانية
- نوع الهدية

قواعد الرد:

- ممنوع HTML
- ممنوع Markdown

إذا أردت عرض صورة استخدم فقط:
IMAGE:رابط_الصورة
"""
        }

    ]

    # =========================
    # 🧠 CLEAN MEMORY
    # =========================

    clean_history = []

    last_answers = set()

    for msg in history[-6:]:

        content = msg["content"].strip()

        # تجاهل الردود القصيرة
        if len(content) < 10:
            continue

        # تجاهل التكرار
        if content in last_answers:
            continue

        # تجاهل الردود السيئة
        bad_answers = [
            "تفضل",
            "تفضل 🌸",
            "أكيد",
            "نعم"
        ]

        if content in bad_answers:
            continue

        last_answers.add(content)

        clean_history.append(msg)

    messages.extend(clean_history)

    # =========================
    # 📦 CONTEXT
    # =========================

    messages.append({
        "role": "system",
        "content": f"""
معلومات قاعدة البيانات:

{context}
"""
    })

    # =========================
    # 👤 USER MESSAGE
    # =========================

    messages.append({
        "role": "user",
        "content": question
    })

    # =========================
    # 💾 SAVE USER
    # =========================

    history.append({
        "role": "user",
        "content": question
    })

    # =========================
    # 🤖 OPENAI RESPONSE
    # =========================

    response = ai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=1.2,
        presence_penalty=0.8,
        frequency_penalty=0.7,
        max_tokens=300
    )

    answer = response.choices[0].message.content.strip()

    # =========================
    # 🚫 BAD ANSWERS
    # =========================

    bad_words = [
        "تفضل",
        "تفضل 🌸",
        "أكيد",
        "نعم"
    ]

    if (
        answer in bad_words
        or len(answer) < 15
    ):

        random_questions = [

            "أكيد 😊 لأي مناسبة بدك الهدية؟",

            "شو نوع البوكيه اللي ببالك؟ 🌸",

            "بتحب يكون التصميم ناعم ولا فخم؟",

            "كم الميزانية اللي حابب تكون تقريباً؟",

            "لمين الهدية؟ 😊"
        ]

        import random

        answer = random.choice(
            random_questions
        )

    # =========================
    # 💾 SAVE ASSISTANT
    # =========================

    history.append({
        "role": "assistant",
        "content": answer
    })

    # =========================
    # ✂️ LIMIT MEMORY
    # =========================

    if len(history) > 12:
        chat_memory[data.session_id] = history[-12:]

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
