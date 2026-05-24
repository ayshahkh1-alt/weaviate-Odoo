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

            # ✅ IMAGE
            "image_url": flower.image_url,

            # ✅ PRODUCT PAGE
            "product_url": flower.product_url,

            "text": f"""
اسم المنتج: {flower.name}

اللون: {flower.color}

السعر: {flower.price}

الكمية المتوفرة: {flower.available}

رابط المنتج:
{flower.product_url}

رابط الصورة:
{flower.image_url}
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
# 🤖 CHAT
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

رابط المنتج:
{props.get("product_url")}

الصورة الجاهزة للعرض:

<a href="{props.get('product_url')}" target="_blank">
    <img 
        src="{props.get('image_url')}" 
        alt="{props.get('name')}"
        width="250"
        style="border-radius:10px;"
    />
</a>

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
أنت مساعد ذكي ومتخصص لمتجر زهور وهدايا.

المعلومات التي تصلك تأتي من:
- منتجات متجر حقيقية
- مقالات قاعدة المعرفة

تعليمات مهمة جداً:

1- فهم سؤال المستخدم بدقة.

2- إذا كان السؤال غير واضح:
اسأل سؤال توضيحي قبل الإجابة.

3- إذا كان السؤال عن منتج:
اذكر:
- اسم المنتج
- السعر
- اللون
- التوفر

4- إذا وجدت صورة أو رابط منتج:
اعرض الصورة مباشرة باستخدام HTML فقط.

5- ممنوع استخدام Markdown links نهائياً.

6- استخدم دائماً هذا الشكل:

<a href="PRODUCT_URL" target="_blank">
   <img src="IMAGE_URL" width="250"/>
</a>

7- لا تقل:
- raw_data
- context

8- إذا لم تجد معلومة:
قل ذلك بوضوح.

9- كن ودوداً وكأنك موظف متجر حقيقي.

10- إذا وجدت عدة منتجات:
رتبها بشكل جميل وواضح.

11- لا تطبع روابط الصور كنص.
اعرض الصور مباشرة.

12- لا تشرح HTML.
فقط استخدمه داخل الإجابة.
لا تعرض المنتجات كنص طويل.

اكتب وصف قصير فقط.

المنتجات سيتم عرضها تلقائياً بالكروت.
"""
            },

            {
                "role": "user",

                "content": f"""
سؤال المستخدم:
{data.question}

المعلومات المتوفرة:
{context}
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