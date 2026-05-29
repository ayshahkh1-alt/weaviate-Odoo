
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

نوع المنتج الحقيقي: {flower.category}

اسم المنتج الحقيقي: {flower.name}

لون المنتج الحقيقي: {flower.color}

السعر الحقيقي: {flower.price}

التوفر الحقيقي: {flower.available}

رابط المنتج الحقيقي: {flower.product_url}

مهم:
لا تغيّر نوع المنتج.
إذا كان المنتج وردة لا تعتبره بوكيه.
إذا كان المنتج بوكيه لا تعتبره وردة.

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
    # 🧠 CLEAN QUESTION
    # =========================

    question = data.question.strip().lower()

    # =========================
    # 💾 SAVE USER MESSAGE
    # =========================

    chat_memory[data.session_id].append({

        "role": "user",

        "content": question
    })

    # =========================
    # 🔍 SEARCH
    # =========================

    results = collection.query.hybrid(

        query=question,

        limit=3
    )

    # =========================
    # 📦 BUILD CONTEXT
    # =========================

    context = ""

    used_products = []

    for obj in results.objects:

        props = obj.properties

        # =========================
        # 🌸 PRODUCT
        # =========================

        if props.get("type") == "product":

            product_name = props.get("name")

            # منع التكرار

            if product_name in used_products:
                continue

            used_products.append(product_name)

            context += f"""

[منتج]

نوع المنتج الحقيقي:
{props.get("category")}

اسم المنتج الحقيقي:
{props.get("name")}

لون المنتج الحقيقي:
{props.get("color")}

السعر الحقيقي:
{props.get("price")}

التوفر الحقيقي:
{props.get("available")}

رابط المنتج:
{props.get("product_url")}

الصورة:
IMAGE:{props.get("image_url")}

"""

        # =========================
        # 📚 ARTICLE
        # =========================

        elif props.get("type") == "article":

            context += f"""

[مقال]

العنوان:
{props.get("title")}

المحتوى:
{props.get("content")}

"""

    # =========================
    # 🧠 BUILD MESSAGES
    # =========================

    messages = [

        {
            "role": "system",
            "content": """

أنت مساعد مبيعات ذكي لمتجر زهور وهدايا.

قواعد صارمة جداً:

- اعتمد فقط على المنتجات الموجودة بالبيانات
- ممنوع اختراع منتجات غير موجودة
- ممنوع تغيير اسم المنتج
- ممنوع تغيير نوع المنتج
- إذا كان النوع وردة لا تقل بوكيه
- إذا كان النوع بوكيه لا تقل وردة
- اعرض اسم المنتج الحقيقي كما هو تماماً
- إذا طلب المستخدم تغيير اللون حافظ على نفس نوع المنتج
- إذا كانت المحادثة عن خطبة ابقِ ضمن اقتراحات الخطبة
- إذا كانت المحادثة عن تخرج ابقِ ضمن اقتراحات التخرج
- تذكر سياق المحادثة السابقة دائماً
- لا تستخدم HTML
- لا تستخدم Markdown
- لا تضع نجوم أو تنسيقات
- لا تكرر نفس المنتج أكثر من مرة
- لا تعرض منتجات لا علاقة لها بالسؤال
- كن واضحاً ومختصراً

طريقة عرض الصور:

IMAGE:رابط_الصورة

إذا كان هناك رابط منتج ضعه كما هو.

اعرض المنتجات بهذا الشكل:

1. اسم المنتج
السعر: 100 شيكل
اللون: أبيض
IMAGE:رابط_الصورة
رابط المنتج: الرابط

كن لطيفاً وكأنك موظف مبيعات محترف.

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
    # 🧠 CURRENT QUESTION
    # =========================

    messages.append({

        "role": "user",

        "content": f"""

سؤال المستخدم:

{question}


البيانات المتوفرة:

{context}

"""
    })

    # =========================
    # 🤖 OPENAI RESPONSE
    # =========================

    response = ai_client.chat.completions.create(

        model="gpt-4o-mini",

        messages=messages,

        temperature=0.4
    )

    assistant_reply = (
        response
        .choices[0]
        .message
        .content
    )

    # =========================
    # 💾 SAVE ASSISTANT REPLY
    # =========================

    chat_memory[data.session_id].append({

        "role": "assistant",

        "content": assistant_reply
    })

    # =========================
    # 🧠 LIMIT MEMORY
    # =========================

    if len(chat_memory[data.session_id]) > 12:

        chat_memory[data.session_id] = (
            chat_memory[data.session_id][-12:]
        )

    # =========================
    # 🛍 PRODUCTS RESPONSE
    # =========================

    products_response = []

    added_products = []

    for obj in results.objects:

        props = obj.properties

        if props.get("type") != "product":
            continue

        product_name = props.get("name")

        if product_name in added_products:
            continue

        added_products.append(product_name)

        products_response.append({

            "name": props.get("name"),

            "color": props.get("color"),

            "category": props.get("category"),

            "price": props.get("price"),

            "available": props.get("available"),

            "image_url": props.get("image_url"),

            "product_url": props.get("product_url")
        })

    # =========================
    # ✅ RESPONSE
    # =========================

    return {

        "answer": assistant_reply,

        "products": products_response
    }


# =========================
# 🔚 CLOSE CONNECTION
# =========================

@app.on_event("shutdown")
def shutdown_event():

    client.close()
