from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
from dotenv import load_dotenv

import weaviate
from weaviate.classes.init import Auth

from openai import OpenAI
import json

load_dotenv()

app = FastAPI()

# =========================
# CORS
# =========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Weaviate
# =========================
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
)

collection = client.collections.get("products")

# =========================
# OpenAI
# =========================
ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# =========================
# MEMORY
# =========================
chat_memory = {}

class Question(BaseModel):
    session_id: str
    question: str


# =========================
# MEMORY FUNCTION
# =========================
def build_memory(session_id, msg):
    if session_id not in chat_memory:
        chat_memory[session_id] = []

    chat_memory[session_id].append({
        "role": "user",
        "content": msg
    })

    return chat_memory[session_id][-10:]


# =========================
# CHAT ENDPOINT
# =========================
@app.post("/chat")
def chat(data: Question):

    # =========================
    # 🧠 MEMORY
    # =========================
    history = build_memory(data.session_id, data.question)

    # =========================
    # 🎯 INTENT EXTRACTION (JSON)
    # =========================
    intent_response = ai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": """
استخرج فقط JSON:

{
  "occasion": null,
  "color": null,
  "product_type": null
}
"""
            },
            {
                "role": "user",
                "content": str(history)
            }
        ]
    )

    try:
        intent = json.loads(intent_response.choices[0].message.content)
    except:
        intent = {"occasion": None, "color": None, "product_type": None}

    # =========================
    # 🔍 SEARCH QUERY
    # =========================
    query = data.question

    if intent.get("occasion"):
        query += f" {intent['occasion']}"

    if intent.get("color"):
        query += f" {intent['color']}"

    if intent.get("product_type"):
        query += f" {intent['product_type']}"

    # =========================
    # WEAVIATE SEARCH
    # =========================
    results = collection.query.near_text(
        query=query,
        limit=6
    )

    context = ""
    products = []

    for obj in results.objects:
        p = obj.properties

        # =========================
        # COLOR FILTER (IMPORTANT)
        # =========================
        if intent.get("color"):
            if p.get("color") and intent["color"].lower() not in p["color"].lower():
                continue

        products.append(p)

        context += f"""
اسم المنتج: {p.get('name')}
اللون: {p.get('color')}
السعر: {p.get('price')}
الكمية: {p.get('available')}
"""

    # =========================
    # 🤖 FINAL ANSWER
    # =========================
    response = ai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[

            # =========================
            # SYSTEM PROMPT (FULL RULES)
            # =========================
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
- واقترح أفضل الخيارات المناسبة

4- إذا كان السؤال يعتمد على مقالات قاعدة المعرفة:
لا تنسخ النص حرفيًا.
استخرج المعلومة المهمة ثم اشرحها بأسلوب طبيعي ومختصر وواضح.

5- إذا لم تجد معلومات كافية:
قل ذلك بوضوح ولا تخترع معلومات.

6- كن ودودًا ولطيفًا وكأنك موظف متجر حقيقي.

7- إذا وُجدت عدة خيارات:
رتبها بشكل مرتب وسهل القراءة.

8- لا تذكر كلمات مثل:
raw_data
context

9- إذا كان السؤال عن مناسبة:
اقترح منتجات مناسبة حسب المناسبة.

10- إذا كان المستخدم محتار:
ساعده بأسئلة ذكية حتى تصل للخيار المناسب.

11- لا تقل أنك لا تستطيع عرض الصور.

12- إذا كانت هناك صورة ضمن المنتج:
اعرضها باستخدام HTML img tag.

13- التزم فقط بالمعلومات الموجودة في البيانات التي تصلك من النظام.
لا تخترع منتجات أو أسعار أو ألوان غير موجودة.
"""
            },

            # =========================
            # USER MESSAGE
            # =========================
            {
                "role": "user",
                "content": f"""
المحادثة السابقة:
{history}

المنتجات المتاحة:
{context}

سؤال المستخدم:
{data.question}
"""
            }
        ]
    )

    answer = response.choices[0].message.content

    # =========================
    # SAVE MEMORY
    # =========================
    chat_memory[data.session_id].append({
        "role": "assistant",
        "content": answer
    })

    # =========================
    # RESPONSE
    # =========================
    return {
        "answer": answer,

        "products": [
            {
                "name": p.get("name"),
                "price": p.get("price"),
                "color": p.get("color"),
                "image_url": p.get("image_url"),
                "orderable": True
            }
            for p in products
        ]
    }


# =========================
# CLOSE CONNECTION
# =========================
@app.on_event("shutdown")
def shutdown():
    client.close()