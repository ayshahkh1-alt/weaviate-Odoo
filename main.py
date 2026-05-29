
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
    category: str
    price: float
    available: float
    image_url: str
    product_url: str


class Article(BaseModel):
    title: str
    content: str


class Question(BaseModel):
    question: str
    session_id: str


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

            "category": flower.category,

            "price": flower.price,

            "available": flower.available,

            "image_url": flower.image_url,

            "product_url": flower.product_url,

            "text": f"""
نوع المنتج: {flower.category}

اسم المنتج: {flower.name}

لون المنتج: {flower.color}

السعر: {flower.price}

التوفر: {flower.available}

رابط المنتج: {flower.product_url}

هذا المنتج مناسب للهدايا والمناسبات المختلفة
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
    # 🧠 CREATE SESSION
    # =========================

    if data.session_id not in chat_memory:

        chat_memory[data.session_id] = []

    # =========================
    # 💾 SAVE USER MESSAGE
    # =========================

    chat_memory[data.session_id].append({
        "role": "user",
        "content": data.question
    })

    # =========================
    # 🔍 SEARCH
    # =========================

    results = collection.query.hybrid(
        query=data.question,
        limit=4
    )

    # =========================
    # 📦 BUILD CONTEXT
    # =========================

    context = ""

    for obj in results.objects:

        props = obj.properties

        # =========================
        # 🌸 PRODUCT
        # =========================

        if props.get("type") == "product":

            context += f"""

[منتج]

النوع: {props.get("category")}

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
    # 🧠 BUILD MESSAGES
    # =========================

    messages = [

        {
            "role": "system",
            "content": """
أنت مساعد ذكي لمتجر زهور وهدايا.

تعليمات مهمة جداً:

- تذكر سياق المحادثة السابقة
- إذا طلب المستخدم تغيير اللون لا تغيّر نوع المنتج
- إذا كانت المحادثة عن خطبة ابقِ ضمن اقتراحات الخطبة
- إذا كانت المحادثة عن تخرج ابقِ ضمن اقتراحات التخرج
- إذا طلب المستخدم وردة مفردة لا تعرض بوكيهات
- إذا طلب المستخدم بوكيه لا تعرض وردة مفردة
- لا تخترع منتجات غير موجودة
- اعتمد فقط على البيانات المرسلة لك
- لا تستخدم HTML نهائياً
- لا تستخدم Markdown
- اكتب بشكل مرتب وواضح
- كن لطيفاً وكأنك موظف مبيعات حقيقي

طريقة عرض الصور:

IMAGE:رابط_الصورة

يمكنك وضع الصورة داخل النص.

إذا يوجد أكثر من منتج اعرضهم بشكل مرتب.
"""
        }
    ]

    # =========================
    # 🧠 ADD PREVIOUS CHAT
    # =========================

    messages.extend(
        chat_memory[data.session_id][-10:]
    )

    # =========================
    # 🧠 ADD CURRENT QUESTION
    # =========================

    messages.append({

        "role": "user",

        "content": f"""

سؤال المستخدم:

{data.question}


البيانات المتوفرة:

{context}

"""
    })

    # =========================
    # 🤖 OPENAI RESPONSE
    # =========================

    response = ai_client.chat.completions.create(

        model="gpt-4o-mini",

        messages=messages
    )

    assistant_reply = response.choices[0].message.content

    # =========================
    # 💾 SAVE ASSISTANT REPLY
    # =========================

    chat_memory[data.session_id].append({
        "role": "assistant",
        "content": assistant_reply
    })

    # =========================
    # ✅ RESPONSE
    # =========================

    return {

        "answer": assistant_reply,

        "products": [

            {

                "name": obj.properties.get("name"),

                "color": obj.properties.get("color"),

                "category": obj.properties.get("category"),

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
