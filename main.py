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
# WEAVIATE CONNECTION
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
# OPENAI CLIENT
# =========================

ai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# =========================
# CHAT MEMORY
# =========================

chat_memory = {}

# =========================
# MODELS
# =========================

class Article(BaseModel):
    title: str
    content: str


class Question(BaseModel):
    question: str
    session_id: str


# =========================
# SAVE ARTICLE ONLY
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
# CHAT ENDPOINT (RAG ONLY)
# =========================

@app.post("/chat")
def chat(data: Question):

    # =========================
    # SESSION
    # =========================

    if data.session_id not in chat_memory:
        chat_memory[data.session_id] = []

    question = data.question.strip().lower()

    chat_memory[data.session_id].append({
        "role": "user",
        "content": question
    })

    # =========================
    # SEARCH ARTICLES ONLY
    # =========================

    results = collection.query.hybrid(
        query=question,
        limit=10,
        alpha=0.3
    )

    # =========================
    # BUILD CONTEXT
    # =========================

    context = ""

    used_articles = []

    for obj in results.objects:
        props = obj.properties

        if props.get("type") != "article":
            continue

        title = props.get("title")

        if title in used_articles:
            continue

        used_articles.append(title)

        context += f"""
[مقال]
العنوان: {title}
المحتوى: {props.get("content")}
"""

    # =========================
    # SYSTEM PROMPT
    # =========================

    messages = [
        {
            "role": "system",
            "content": """
أنت مساعد ذكي يعتمد فقط على المقالات المتوفرة.

قواعد صارمة:

- استخدم فقط المعلومات الموجودة في المقالات
- ممنوع اختلاق معلومات غير موجودة
- إذا لم تجد إجابة قل: لا توجد معلومات كافية
- كن دقيقاً وواضحاً
- لا تستخدم HTML
- لا تستخدم Markdown
- لا تستخدم تنسيقات أو نجوم
- اجعل الإجابة مفهومة وبسيطة
لا تخترع منتجات جديدة 
ممنوع استخدام صور لمسكات
"""
        }
    ]

    # =========================
    # MEMORY
    # =========================

    messages.extend(chat_memory[data.session_id][-10:])

    # =========================
    # USER INPUT + CONTEXT
    # =========================

    messages.append({
        "role": "user",
        "content": f"""
سؤال المستخدم:
{question}

المقالات المتوفرة:
{context}
"""
    })

    # =========================
    # OPENAI RESPONSE
    # =========================

    response = ai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0.3
    )

    answer = response.choices[0].message.content

    # =========================
    # SAVE MEMORY
    # =========================

    chat_memory[data.session_id].append({
        "role": "assistant",
        "content": answer
    })

    if len(chat_memory[data.session_id]) > 12:
        chat_memory[data.session_id] = chat_memory[data.session_id][-12:]

    # =========================
    # RESPONSE
    # =========================

    return {
        "answer": answer
    }


# =========================
# SHUTDOWN
# =========================

@app.on_event("shutdown")
def shutdown_event():
    client.close()