import streamlit as st
import pandas as pd
import sqlite3
import joblib
import matplotlib.pyplot as plt
from datetime import datetime, date
from pathlib import Path

# -------------------- App setup --------------------
st.set_page_config(page_title="Retail Management System", page_icon="🛍️", layout="wide")

DB_PATH = Path(__file__).with_name("retail_store.db")
BASE_DIR = Path(__file__).resolve().parent

# Readable, consistent styling
st.markdown("""
<style>
:root { color-scheme: light; }
.stApp { background: #f5f7fb; color: #172033; }
[data-testid="stSidebar"] { background: #101827; }
[data-testid="stSidebar"] * { color: #f3f6fc !important; }
h1, h2, h3, p, label, [data-testid="stMetricLabel"] { color: #172033; }
.hero { padding: 1.35rem 1.6rem; border-radius: 18px; color: white;
        background: linear-gradient(115deg,#15264b,#2457d6); margin-bottom: 1rem; }
.hero h1, .hero p { color: white !important; }
.small-muted { color: #667085 !important; }
div[data-testid="stMetric"] { background: white; border: 1px solid #e4e9f2;
    border-radius: 14px; padding: 14px 16px; }
div[data-testid="stForm"] { background: white; padding: 16px; border-radius: 14px; border: 1px solid #e4e9f2; }
</style>
""", unsafe_allow_html=True)

# -------------------- SQLite --------------------
def get_conn():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def init_db():
    with get_conn() as conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS products (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            category TEXT NOT NULL,
            price REAL NOT NULL CHECK(price >= 0),
            stock INTEGER NOT NULL CHECK(stock >= 0),
            reorder_level INTEGER NOT NULL DEFAULT 10,
            updated_at TEXT NOT NULL
        )""")
        conn.execute("""CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id INTEGER NOT NULL,
            product_name TEXT NOT NULL,
            quantity INTEGER NOT NULL,
            unit_price REAL NOT NULL,
            total REAL NOT NULL,
            sold_at TEXT NOT NULL,
            FOREIGN KEY(product_id) REFERENCES products(id)
        )""")

def read_products():
    with get_conn() as conn:
        return pd.read_sql_query("SELECT * FROM products ORDER BY id DESC", conn)

def read_sales():
    with get_conn() as conn:
        return pd.read_sql_query("SELECT * FROM sales ORDER BY sold_at DESC", conn)

init_db()

# -------------------- Sidebar --------------------
st.sidebar.markdown("# 🛍️ Retail Intelligence")
st.sidebar.caption("Retail Management System")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "WORKSPACE",
    ["Overview", "Inventory", "Record a Sale", "Sales & Reports", "ML Predictions", "Model Information"],
    index=0
)
st.sidebar.markdown("---")
st.sidebar.caption("Local database")
st.sidebar.write("Your products and sales are saved on this computer.")
if st.sidebar.button("Refresh data", use_container_width=True):
    st.rerun()

products = read_products()
sales = read_sales()

# -------------------- Overview --------------------
if page == "Overview":
    st.markdown("""
    <div class="hero">
      <h1>Retail Management System</h1>
      <p>Manage products, monitor stock, record sales, and explore machine-learning predictions.</p>
    </div>
    """, unsafe_allow_html=True)

    total_products = len(products)
    total_units = int(products["stock"].sum()) if not products.empty else 0
    sales_total = float(sales["total"].sum()) if not sales.empty else 0.0
    low_count = int((products["stock"] <= products["reorder_level"]).sum()) if not products.empty else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Products", f"{total_products:,}")
    c2.metric("Units in stock", f"{total_units:,}")
    c3.metric("Recorded sales", f"₹{sales_total:,.2f}")
    c4.metric("Low-stock items", f"{low_count:,}")

    st.subheader("Quick actions")
    q1, q2, q3 = st.columns(3)
    with q1:
        st.info("**Inventory**\n\nAdd products, edit prices, and adjust quantities.")
    with q2:
        st.info("**Record a Sale**\n\nSave a transaction and automatically deduct stock.")
    with q3:
        st.info("**Sales & Reports**\n\nReview transaction history and export CSV reports.")

    left, right = st.columns([1.4, 1])
    with left:
        st.subheader("Recent sales")
        if sales.empty:
            st.caption("No sales recorded yet. Use **Record a Sale** to add your first transaction.")
        else:
            st.dataframe(sales.head(8)[["sold_at", "product_name", "quantity", "unit_price", "total"]],
                         use_container_width=True, hide_index=True)
    with right:
        st.subheader("Stock watch")
        if products.empty:
            st.caption("No products yet. Open Inventory and add your products.")
        else:
            low = products[products["stock"] <= products["reorder_level"]]
            if low.empty:
                st.success("No low-stock alerts right now.")
            else:
                st.warning(f"{len(low)} product(s) at or below reorder level.")
                st.dataframe(low[["name", "stock", "reorder_level"]], use_container_width=True, hide_index=True)

    if not sales.empty:
        st.subheader("Sales trend")
        chart_data = sales.copy()
        chart_data["day"] = pd.to_datetime(chart_data["sold_at"], errors="coerce").dt.date
        daily = chart_data.groupby("day", as_index=False)["total"].sum().sort_values("day")
        if not daily.empty:
            st.line_chart(daily.set_index("day")["total"], y_label="Sales (₹)")

# -------------------- Inventory --------------------
elif page == "Inventory":
    st.title("📦 Inventory Management")
    st.write("Add and manage product records. Changes are saved in the local SQLite database.")
    with st.expander("➕ Add a new product", expanded=products.empty):
        with st.form("add_product_form", clear_on_submit=True):
            a, b = st.columns(2)
            name = a.text_input("Product name *", placeholder="e.g. Wireless Mouse")
            category = b.text_input("Category *", placeholder="e.g. Electronics")
            c, d, e = st.columns(3)
            price = c.number_input("Selling price (₹)", min_value=0.0, value=100.0, step=10.0)
            stock = d.number_input("Opening stock", min_value=0, value=10, step=1)
            reorder = e.number_input("Low-stock threshold", min_value=0, value=10, step=1)
            submitted = st.form_submit_button("Add product", type="primary")
            if submitted:
                if not name.strip() or not category.strip():
                    st.error("Enter both product name and category.")
                else:
                    with get_conn() as conn:
                        conn.execute(
                            "INSERT INTO products(name,category,price,stock,reorder_level,updated_at) VALUES(?,?,?,?,?,?)",
                            (name.strip(), category.strip(), float(price), int(stock), int(reorder), datetime.now().isoformat(timespec="seconds"))
                        )
                    st.success(f"Added {name.strip()}.")
                    st.rerun()

    products = read_products()
    if products.empty:
        st.info("Inventory is empty. Add your first product above.")
    else:
        f1, f2 = st.columns([2, 1])
        search = f1.text_input("Search products", placeholder="Search by product or category")
        cats = ["All categories"] + sorted(products["category"].dropna().astype(str).unique().tolist())
        cat = f2.selectbox("Category filter", cats)
        filtered = products.copy()
        if search:
            filtered = filtered[
                filtered["name"].str.contains(search, case=False, na=False) |
                filtered["category"].str.contains(search, case=False, na=False)
            ]
        if cat != "All categories":
            filtered = filtered[filtered["category"] == cat]
        display = filtered.copy()
        display["stock_status"] = display.apply(
            lambda r: "LOW STOCK" if r["stock"] <= r["reorder_level"] else "In stock", axis=1
        )
        st.dataframe(display[["id", "name", "category", "price", "stock", "reorder_level", "stock_status", "updated_at"]],
                     use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export inventory CSV", filtered.to_csv(index=False).encode("utf-8"),
                           "inventory_report.csv", "text/csv")
        st.markdown("---")
        st.subheader("Edit or delete a product")
        product_options = {f'#{int(r.id)} — {r["name"]} (stock: {int(r["stock"])})': int(r["id"])
                           for _, r in products.iterrows()}
        chosen_label = st.selectbox("Choose product", list(product_options.keys()))
        chosen_id = product_options[chosen_label]
        chosen = products[products["id"] == chosen_id].iloc[0]
        with st.form("edit_product_form"):
            x1, x2 = st.columns(2)
            edit_name = x1.text_input("Product name", value=str(chosen["name"]))
            edit_cat = x2.text_input("Category", value=str(chosen["category"]))
            x3, x4, x5 = st.columns(3)
            edit_price = x3.number_input("Price (₹)", min_value=0.0, value=float(chosen["price"]), step=10.0, key="edit_price")
            edit_stock = x4.number_input("Stock quantity", min_value=0, value=int(chosen["stock"]), step=1, key="edit_stock")
            edit_reorder = x5.number_input("Reorder level", min_value=0, value=int(chosen["reorder_level"]), step=1, key="edit_reorder")
            u1, u2 = st.columns(2)
            save = u1.form_submit_button("Save changes", type="primary")
            delete = u2.form_submit_button("Delete product")
            if save:
                if not edit_name.strip() or not edit_cat.strip():
                    st.error("Product name and category cannot be empty.")
                else:
                    with get_conn() as conn:
                        conn.execute("UPDATE products SET name=?,category=?,price=?,stock=?,reorder_level=?,updated_at=? WHERE id=?",
                                     (edit_name.strip(), edit_cat.strip(), float(edit_price), int(edit_stock),
                                      int(edit_reorder), datetime.now().isoformat(timespec="seconds"), chosen_id))
                    st.success("Product updated.")
                    st.rerun()
            if delete:
                with get_conn() as conn:
                    conn.execute("DELETE FROM products WHERE id=?", (chosen_id,))
                st.success("Product deleted.")
                st.rerun()

# -------------------- Record a sale --------------------
elif page == "Record a Sale":
    st.title("🧾 Record a Sale")
    st.write("Save a sale transaction. Stock is deducted automatically after validation.")
    products = read_products()
    if products.empty:
        st.warning("Add products in Inventory before recording a sale.")
    else:
        available = products[products["stock"] > 0]
        if available.empty:
            st.warning("There are no products with available stock.")
        else:
            options = {f'{r["name"]} — ₹{float(r["price"]):,.2f} (stock {int(r["stock"])})': int(r["id"])
                       for _, r in available.iterrows()}
            with st.form("sale_form"):
                selected = st.selectbox("Product", list(options.keys()))
                product_id = options[selected]
                prod = available[available["id"] == product_id].iloc[0]
                quantity = st.number_input("Quantity sold", min_value=1, max_value=int(prod["stock"]), value=1, step=1)
                unit_price = st.number_input("Unit selling price (₹)", min_value=0.0, value=float(prod["price"]), step=10.0)
                sale_date = st.date_input("Sale date", value=date.today())
                confirm = st.form_submit_button("Save sale", type="primary")
                if confirm:
                    total = float(quantity) * float(unit_price)
                    timestamp = datetime.combine(sale_date, datetime.now().time()).isoformat(timespec="seconds")
                    with get_conn() as conn:
                        current = conn.execute("SELECT stock,name FROM products WHERE id=?", (product_id,)).fetchone()
                        if current is None or int(current[0]) < int(quantity):
                            st.error("Not enough stock available. Refresh and try again.")
                        else:
                            conn.execute("INSERT INTO sales(product_id,product_name,quantity,unit_price,total,sold_at) VALUES(?,?,?,?,?,?)",
                                         (product_id, current[1], int(quantity), float(unit_price), total, timestamp))
                            conn.execute("UPDATE products SET stock=stock-?, updated_at=? WHERE id=?",
                                         (int(quantity), datetime.now().isoformat(timespec="seconds"), product_id))
                            st.success(f"Sale saved. Total: ₹{total:,.2f}. Stock updated.")
                            st.rerun()

# -------------------- Sales and reports --------------------
elif page == "Sales & Reports":
    st.title("📊 Sales & Reports")
    sales = read_sales()
    products = read_products()
    a, b, c = st.columns(3)
    a.metric("Transactions", len(sales))
    b.metric("Units sold", int(sales["quantity"].sum()) if not sales.empty else 0)
    c.metric("Total sales", f'₹{sales["total"].sum():,.2f}' if not sales.empty else "₹0.00")
    if sales.empty:
        st.info("No transactions yet. Record a sale to generate reports.")
    else:
        sales["sold_at"] = pd.to_datetime(sales["sold_at"], errors="coerce")
        start = st.date_input("From date", value=sales["sold_at"].min().date())
        end = st.date_input("To date", value=date.today())
        mask = sales["sold_at"].dt.date.between(start, end)
        filtered_sales = sales[mask].copy()
        st.subheader("Transaction history")
        st.dataframe(filtered_sales, use_container_width=True, hide_index=True)
        st.download_button("⬇️ Export sales CSV", filtered_sales.to_csv(index=False).encode("utf-8"),
                           "sales_report.csv", "text/csv")
        if not filtered_sales.empty:
            daily = filtered_sales.assign(day=filtered_sales["sold_at"].dt.date).groupby("day")["total"].sum()
            st.subheader("Revenue by day")
            st.bar_chart(daily, y_label="Revenue (₹)")
            st.subheader("Best-selling products")
            best = filtered_sales.groupby("product_name", as_index=False).agg(
                units_sold=("quantity", "sum"), revenue=("total", "sum")
            ).sort_values("units_sold", ascending=False)
            st.dataframe(best, use_container_width=True, hide_index=True)
        st.caption("Sales total is based only on transactions recorded in this app.")

# -------------------- ML predictions --------------------
elif page == "ML Predictions":
    st.title("🤖 Machine Learning Predictions")
    st.write("Upload a CSV containing the features expected by your trained model.")
    uploaded = st.file_uploader("Upload prediction dataset (CSV)", type=["csv"], key="ml_csv")
    model_choice = st.selectbox("Model", ["Logistic Regression", "Random Forest"])
    model_file = "logistic_model.pkl" if model_choice == "Logistic Regression" else "random_forest_model.pkl"
    model_path = BASE_DIR / model_file
    model = None
    if model_path.exists():
        try:
            model = joblib.load(model_path)
            st.success(f"{model_choice} model loaded.")
        except Exception as exc:
            st.error(f"Could not load model: {exc}")
    else:
        st.warning(f"Model file not found: {model_file}. Check the model filename in your project folder.")
    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded)
            st.write(f"Dataset: **{len(df):,} rows × {len(df.columns):,} columns**")
            st.dataframe(df.head(10), use_container_width=True)
            if model is not None:
                features = getattr(model, "feature_names_in_", None)
                if features is None:
                    st.info("This model does not expose feature_names_in_. Check the original training notebook for the exact input columns.")
                else:
                    missing = [f for f in features if f not in df.columns]
                    if missing:
                        st.error("Required feature columns are missing: " + ", ".join(map(str, missing)))
                    else:
                        X = df[list(features)].copy()
                        if hasattr(model, "predict_proba"):
                            pred = model.predict(X)
                            prob = model.predict_proba(X)
                            out = df.copy()
                            out["prediction"] = pred
                            if prob.ndim == 2 and prob.shape[1] >= 2:
                                out["positive_class_probability"] = prob[:, 1]
                            st.subheader("Prediction results")
                            st.dataframe(out, use_container_width=True)
                            st.download_button("⬇️ Download predictions", out.to_csv(index=False).encode("utf-8"),
                                               "ml_predictions.csv", "text/csv")
                        else:
                            out = df.copy()
                            out["prediction"] = model.predict(X)
                            st.dataframe(out, use_container_width=True)
                            st.download_button("⬇️ Download predictions", out.to_csv(index=False).encode("utf-8"),
                                               "ml_predictions.csv", "text/csv")
        except Exception as exc:
            st.error(f"Could not process the CSV: {exc}")

# -------------------- Model information --------------------
elif page == "Model Information":
    st.title("🧠 Model Information")
    st.write("These are previously reported evaluation results from the project. Re-run evaluation on the held-out test set to verify them.")
    metrics = pd.DataFrame([
        {"Model": "Logistic Regression", "Accuracy": "92.14%", "ROC-AUC": "0.9752"},
        {"Model": "Random Forest", "Accuracy": "92.18%", "ROC-AUC": "0.9720"},
    ])
    st.dataframe(metrics, use_container_width=True, hide_index=True)
    st.caption("Reported results are not recalculated by this page.")
    st.markdown("---")
    st.subheader("Model files")
    for filename in ["logistic_model.pkl", "random_forest_model.pkl"]:
        path = BASE_DIR / filename
        if path.exists():
            st.success(f"{filename} — file found")
        else:
            st.warning(f"{filename} — not found")
    st.markdown("""
    **What these metrics mean**
    - **Accuracy:** share of predictions that matched the actual labels on the evaluation set.
    - **ROC-AUC:** how well the model separates the two classes across probability thresholds.
    - For reliable reporting, record the dataset version, test split, target column, and preprocessing steps.
    """)
