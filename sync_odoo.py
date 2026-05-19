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
uid = common.authenticate(db, username, password, {})

models = xmlrpc.client.ServerProxy(f"{url}/xmlrpc/2/object")

print("Connected to Odoo ✔")


# =========================
# 🔗 WEAVIATE CONNECTION
# =========================

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
    headers={
        "X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")
    }
)

collection = client.collections.get("KnowledgeBase")

print("Connected to Weaviate ✔")


# =========================
# 🚫 FILTERS
# =========================

ALLOWED_CATEGORIES = [
    "ورد طبيعي",
    "ورد صناعي",
    "بوكيهات"
]

BLOCKED_KEYWORDS = [
    "مسكات", "مسكة", "عرائس", "كماليات", "اكسسوارات",
    "ورشة", "دورة", "تدريب",
    "delivery", "discount", "gift card",
    "booking", "fees", "wallet", "invoice"
]

# 👇 فلترة احترافية للمقالات
BAD_PATTERNS = [
    "@",
    "api-user",
    "test",
    "مرحبا",
    "gmail.com",
    "stu.najah.edu"
]


# =========================
# 🌸 PRODUCTS
# =========================

products = models.execute_kw(
    db, uid, password,
    'product.template',
    'search_read',
    [[]],
    {
        'fields': ['id', 'name', 'list_price', 'categ_id']
    }
)

print(f"Found {len(products)} products")


for p in products:

    product_name = p["name"].lower()

    category_raw = p.get("categ_id")
    if isinstance(category_raw, list) and len(category_raw) > 1:
        category_name = category_raw[1].strip()
    else:
        category_name = ""

    # ❌ filters
    if any(word in product_name for word in BLOCKED_KEYWORDS):
        continue

    if category_name not in ALLOWED_CATEGORIES:
        continue

    product_id = p["id"]

    variants = models.execute_kw(
        db, uid, password,
        'product.product',
        'search_read',
        [[['product_tmpl_id', '=', product_id]]],
        {
            'fields': ['id', 'name', 'qty_available']
        }
    )

    for v in variants:

        variant_image = f"{url}/web/image/product.product/{v['id']}/image_1920"

        text = safe_text(f"""
        المنتج: {p['name']}
        السعر: {p['list_price']}
        اللون/النوع: {v['name']}
        الكمية: {v['qty_available']}
        """)

        try:
            collection.data.insert(
                properties={
                    "type": "product",
                    "name": p["name"],
                    "color": v["name"],
                    "price": p["list_price"],
                    "available": v["qty_available"],
                    "image_url": variant_image,
                    "text": text
                }
            )

            print(f"Inserted: {p['name']} - {v['name']}")

        except Exception as e:
            print("Skipped product:", str(e)[:200])


# =========================
# 📚 ARTICLES
# =========================

articles = models.execute_kw(
    db, uid, password,
    'knowledge.article',
    'search_read',
    [[]],
    {
        'fields': ['name', 'body']
    }
)

print(f"Found {len(articles)} articles")


for article in articles:

    title = article.get("name") or ""
    title = str(title).strip()

    # ❌ skip empty titles
    if not title:
        continue

    # ❌ advanced filtering
    if any(p in title.lower() for p in BAD_PATTERNS):
        continue

    if len(title.split()) < 2:
        continue

    raw_body = article.get("body", "")
    body = clean_html(raw_body)
    body = safe_text(body, 2000)

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

        print(f"Inserted article: {title}")

    except Exception as e:
        print("Skipped article:", str(e)[:200])


# =========================
# ✅ DONE
# =========================

print("DONE ✔")

client.close()