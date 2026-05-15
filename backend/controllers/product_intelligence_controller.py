import json
import os
import logging
import pandas as pd
from flask import jsonify, request
from pathlib import Path
from typing import Dict

logger = logging.getLogger("product_intelligence_controller")

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DB_FILE  = BASE_DIR / "data" / "products.json"

# ── Lazy-load enricher to avoid circular imports ────────────────────────────
_enricher = None

def _get_enricher():
    global _enricher
    if _enricher is None:
        try:
            import sys
            if str(BASE_DIR) not in sys.path:
                sys.path.insert(0, str(BASE_DIR))
            from data_enrichment import DataEnricher
            _enricher = DataEnricher
            logger.info("DataEnricher loaded successfully.")
        except Exception as exc:
            logger.warning("DataEnricher unavailable — using raw products: %s", exc)
            _enricher = None
    return _enricher


def _load_db():
    if not DB_FILE.exists():
        return []
    try:
        with open(DB_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _save_db(data):
    os.makedirs(DB_FILE.parent, exist_ok=True)
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=2)


def _is_na(val) -> bool:
    """Return True if val is NaN / None / empty."""
    try:
        import math
        if val is None:
            return True
        if isinstance(val, float) and math.isnan(val):
            return True
        if str(val).strip().lower() in ("nan", "none", "", "null", "na"):
            return True
        return False
    except Exception:
        return False


def handle_product_upload(req):
    """
    POST /api/pipeline/upload
    Handles multi-tenant CSV upload. Saves to SQLite and triggers background pipeline.
    """
    from database.models import db, Product, Inventory, Sale, Dealer
    from utils.auth import get_dealer_id_from_token
    
    # 1. Auth check
    auth_header = req.headers.get("Authorization")
    dealer_id = get_dealer_id_from_token(auth_header)
    if not dealer_id:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    if "file" not in req.files:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    file = req.files["file"]
    if file.filename == "":
        return jsonify({"status": "error", "message": "No selected file"}), 400

    try:
        # Load and clean CSV
        df = pd.read_csv(file)
        col_map = {col.lower().strip(): col for col in df.columns}

        # ── Column Mapping ───────────────────────────────────────────────────
        name_col   = (col_map.get("product_name") or col_map.get("name")
                      or col_map.get("product") or col_map.get("item"))
        ing_col    = (col_map.get("ingredients") or col_map.get("ingredient_list")
                      or col_map.get("ingredient"))
        brand_col  = col_map.get("brand") or col_map.get("manufacturer")
        price_col  = (col_map.get("price") or col_map.get("msrp")
                      or col_map.get("unit_price"))
        sales_col  = (col_map.get("units_sold") or col_map.get("sales")
                      or col_map.get("volume") or col_map.get("units"))
        id_col     = col_map.get("product_id") or col_map.get("id")

        if not name_col:
            return jsonify({
                "status": "error",
                "message": "CSV must contain at least a 'product_name' or 'name' column"
            }), 400

        # ── Process Rows ─────────────────────────────────────────────────────
        base_ts = int(pd.Timestamp.now().timestamp() * 1000)
        new_products_added = 0
        preview = []

        for idx, row in df.iterrows():
            raw_id = (str(row[id_col]) if id_col and not _is_na(row[id_col]) else f"UP_{base_ts}_{idx}")
            p_name = str(row[name_col])
            
            # Check if product already exists for this dealer to avoid unique constraint errors
            existing = Product.query.filter_by(product_id=raw_id, dealer_id=dealer_id).first()
            if existing:
                continue

            # Create Product
            new_p = Product(
                product_id=raw_id,
                dealer_id=dealer_id,
                product_name=p_name,
                brand=(str(row[brand_col]) if brand_col and not _is_na(row[brand_col]) else "Unknown"),
                price=(float(row[price_col]) if price_col and not _is_na(row[price_col]) else 25.0),
                ingredients=(str(row[ing_col]) if ing_col and not _is_na(row[ing_col]) else ""),
            )
            db.session.add(new_p)

            # Create Inventory placeholder
            new_inv = Inventory(
                product_id=raw_id,
                dealer_id=dealer_id,
                current_stock=100.0,
                reorder_level=30.0
            )
            db.session.add(new_inv)

            # Create Sales placeholder (or real if in CSV)
            units = (float(row[sales_col]) if sales_col and not _is_na(row[sales_col]) else 50.0)
            new_sale = Sale(
                product_id=raw_id,
                dealer_id=dealer_id,
                units_sold=units,
                sale_date=pd.Timestamp.now()
            )
            db.session.add(new_sale)

            new_products_added += 1

            # Add to preview (limit 10)
            if len(preview) < 10:
                preview.append({
                    "product_id": raw_id,
                    "product_name": p_name,
                    "brand": new_p.brand,
                    "price": new_p.price
                })

        db.session.commit()

        # ── Trigger AI Pipeline ──────────────────────────────────────────────
        if new_products_added > 0:
            import threading
            from routes.auth_routes import _run_pipeline_safe
            thread = threading.Thread(target=_run_pipeline_safe, args=(dealer_id,), daemon=True)
            thread.start()

        return jsonify({
            "status": "success",
            "message": f"Successfully uploaded {new_products_added} products. AI analysis started.",
            "data": {
                "uploaded_count": new_products_added,
                "preview": preview,
                "filename": file.filename,
                "pipeline_running": True
            }
        })

    except Exception as exc:
        db.session.rollback()
        logger.exception("Upload handler fatal error")
        return jsonify({"status": "error", "message": str(exc)}), 500


def search_products(q):
    db = _load_db()
    if not q:
        return jsonify({"status": "success", "data": db[:10]})
    q = q.lower()
    matches = [
        p for p in db
        if q in p.get("product_name", "").lower()
        or q in p.get("ingredients", "").lower()
        or q in p.get("brand", "").lower()
    ]
    return jsonify({"status": "success", "data": matches})


def get_all_products():
    return _load_db()


def get_paginated_products(page=1, limit=20, search="", has_ingredients=False):
    all_products = get_all_products()

    filtered = all_products
    if search:
        q = search.lower()
        filtered = [
            p for p in filtered
            if q in p.get("product_name", "").lower()
            or q in p.get("brand", "").lower()
            or q in p.get("category", "").lower()
        ]

    if has_ingredients:
        filtered = [p for p in filtered
                    if p.get("ingredients") and len(str(p["ingredients"])) > 5]

    total = len(filtered)
    start = (page - 1) * limit
    return {
        "products":    filtered[start: start + limit],
        "total":       total,
        "page":        page,
        "limit":       limit,
        "total_pages": max(1, (total + limit - 1) // limit),
    }


def get_dashboard_stats():
    products = get_all_products()
    brands   = {p.get("brand") for p in products
                if p.get("brand") and p.get("brand") != "Unknown"}
    with_ing = sum(1 for p in products
                   if p.get("ingredients") and len(str(p["ingredients"])) > 5)

    # Use enriched category field if available, else infer
    cats: Dict = {}
    for p in products:
        cat = p.get("category")
        if not cat or cat == "general":
            name = p.get("product_name", "").lower()
            if "serum"       in name: cat = "serum"
            elif "cream"     in name or "moisturizer" in name: cat = "cream"
            elif "cleanser"  in name or "wash"        in name: cat = "cleanser"
            elif "sunscreen" in name or "spf"         in name: cat = "sunscreen"
            elif "toner"     in name: cat = "toner"
            elif "mask"      in name: cat = "mask"
            elif "oil"       in name: cat = "oil"
            else: cat = "general"
        cats[cat] = cats.get(cat, 0) + 1

    sorted_cats    = dict(sorted(cats.items(), key=lambda x: x[1], reverse=True)[:6])
    top_candidates = [k for k in sorted_cats if k != "general"]
    top_cat        = top_candidates[0].title() if top_candidates else "Skincare"

    # Safety distribution from pre-computed safety
    safe_count     = sum(1 for p in products
                         if p.get("precomputed_safety", {}).get("status") == "Safe")
    moderate_count = sum(1 for p in products
                         if p.get("precomputed_safety", {}).get("status") == "Moderate")
    unsafe_count   = sum(1 for p in products
                         if p.get("precomputed_safety", {}).get("status") == "Unsafe")

    import random
    rng  = random.Random(len(products))
    base = max(1, len(products)) * 80
    trend_vals = [
        int(base * rng.uniform(0.7, 0.9)),
        int(base * rng.uniform(0.9, 1.1)),
        int(base * rng.uniform(1.1, 1.4)),
        int(base * rng.uniform(1.4, 1.8)),
    ]

    return {
        "total_products":            len(products),
        "total_brands":              len(brands),
        "products_with_ingredients": with_ing,
        "top_category":              top_cat,
        "category_distribution":     sorted_cats,
        "safety_distribution": {
            "safe":     safe_count,
            "moderate": moderate_count,
            "unsafe":   unsafe_count,
        },
        "sales_trend": {
            "labels": ["Week 1", "Week 2", "Week 3", "Week 4"],
            "values": trend_vals,
        },
    }


def clear_db():
    if DB_FILE.exists():
        os.remove(DB_FILE)
    return jsonify({"status": "success", "message": "Database cleared"})


def analyze_stored_product(data, svc):
    prod_id  = data.get("product_id")
    db       = _load_db()
    product  = next((p for p in db if p["product_id"] == prod_id), None)

    if not product:
        product = {
            "product_name": data.get("product_name", "Unknown"),
            "ingredients":  data.get("ingredients", ""),
            "review":       data.get("review", ""),
        }

    results = {}

    # Use enriched skin_text if available for better skin model input
    review_text = (product.get("skin_text") or product.get("review", ""))
    if review_text:
        try:    results["skin"]      = svc.predict_skin(review_text)
        except: results["skin"]      = None
        try:    results["sentiment"] = svc.analyze_sentiment(review_text)
        except: results["sentiment"] = None

    # Use pre-computed safety if available (from enrichment layer)
    if product.get("precomputed_safety"):
        results["safety"] = product["precomputed_safety"]
    else:
        try:
            results["safety"] = svc.detect_harmful(
                ingredient_text=product.get("ingredients", ""),
                product_id=prod_id,
                product_name=product.get("product_name", ""),
            )
        except:
            results["safety"] = None

    return jsonify({"status": "success", "product": product, "analysis": results})
