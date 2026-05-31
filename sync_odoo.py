import os
import xmlrpc.client
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth
from bs4 import BeautifulSoup

load_dotenv()

# =========================
# 🔧 SAFE FUNCTIONS
# =========================

def safe_text(text, max_length=2000):
    if not text:
        return ""
    return str(text)[:max_length]


def clean_html(html):
    if not html:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    text = soup.get_text(separator=" ", strip=True)

    return " ".join(text.split())


# =========================
# 🔗 ODOO CONNECTION
# =========================

url = os.getenv("ODOO_URL")
db = os.getenv("ODOO_DB")
username = os.getenv("ODOO_USERNAME")
password = os.getenv("ODOO_PASSWORD")

common = xmlrpc.client.ServerProxy(f"{url}/xmlrpc/2/common")

uid = common.authenticate(
    db,
    username,
    password,
    {}
)

models = xmlrpc.client.ServerProxy(
    f"{url}/xmlrpc/2/object"
)

print("Connected to Odoo ✔")


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

print("Connected to Weaviate ✔")


# =========================
# 🚫 FILTERS (ARTICLES ONLY)
# =========================

BAD_PATTERNS = [
    "@",
    "api-user",
    "test",
    "gmail.com",
    "stu.najah.edu"
]


# =========================
# 📚 ARTICLES ONLY
# =========================

articles = models.execute_kw(
    db,
    uid,
    password,
    'knowledge.article',
    'search_read',
    [[]],
    {
        'fields': [
            'name',
            'body'
        ]
    }
)

print(f"Found {len(articles)} articles")


for article in articles:

    title = article.get("name") or ""
    title = str(title).strip()

    # ❌ skip empty titles
    if not title:
        continue

    # ❌ filter bad patterns
    if any(p in title.lower() for p in BAD_PATTERNS):
        continue

    raw_body = article.get("body", "")
    body = clean_html(raw_body)
    body = safe_text(body, 2000)

    if not body:
        continue

    text = safe_text(f"""
عنوان المقال:
{title}

المحتوى:
{body}
""")

    try:
        collection.data.insert(
            properties={
                "type": "article",
                "title": title,
                "content": body,
                "text": text
            }
        )

        print(f"Inserted article ✔: {title}")

    except Exception as e:
        print("Skipped article:", str(e)[:200])


# =========================
# ✅ DONE
# =========================

print("DONE ✔")

client.close()