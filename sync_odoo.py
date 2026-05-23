import os
import xmlrpc.client
from dotenv import load_dotenv
import weaviate
from weaviate.classes.init import Auth

load_dotenv()

# =========================
# ODOO
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
# WEAVIATE
# =========================
client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.getenv("WEAVIATE_URL"),
    auth_credentials=Auth.api_key(os.getenv("WEAVIATE_API_KEY")),
    headers={"X-OpenAI-Api-Key": os.getenv("OPENAI_API_KEY")}
)

collection = client.collections.get("products")

print("Connected to Weaviate ✔")

# =========================
# PRODUCTS
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

    product_id = p["id"]
    product_name = p["name"]

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

        image_url = f"{url}/web/image/product.product/{v['id']}/image_1920"
        product_url = f"{url}/shop/product/{product_id}"

        text = f"""
اسم المنتج: {product_name}
النوع: {v['name']}
السعر: {p['list_price']}
الكمية: {v['qty_available']}
"""

        try:
            collection.data.insert({
                "name": product_name,
                "color": v["name"],
                "price": p["list_price"],
                "available": v["qty_available"],
                "image_url": image_url,
                "product_url": product_url,
                "text": text
            })

            print(f"Inserted ✔ {product_name} - {v['name']}")

        except Exception as e:
            print("Skipped:", str(e)[:200])

client.close()
print("DONE ✔")