"""
pipeline_routes.py — Biagiotti Pipeline Integration API  v3
=============================================================
Endpoints:
  GET  /api/pipeline/stats
  GET  /api/pipeline/products
  GET  /api/pipeline/product/<id>
  POST /api/pipeline/upload
  GET  /api/pipeline/report/<id>
  POST /api/pipeline/smart-similarity
  POST /api/pipeline/similar-by-ingredients
"""

from __future__ import annotations

import csv
import io
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Blueprint, jsonify, request

logger = logging.getLogger("pipeline_routes")

pipeline_bp = Blueprint("pipeline", __name__)

# ─── Paths ─────────────────────────────────────────────────────────────────────
_BACKEND_DIR  = Path(__file__).resolve().parent.parent
_CLEANED_DIR  = _BACKEND_DIR / "data" / "cleaned"
_PRODUCTS_CSV = _CLEANED_DIR / "master_products_cleaned.csv"
_SALES_CSV    = _CLEANED_DIR / "master_sales_cleaned.csv"
_REVIEWS_CSV  = _CLEANED_DIR / "master_reviews_cleaned.csv"

# ─── In-memory upload store ────────────────────────────────────────────────────
_UPLOADED_PRODUCTS: List[Dict] = []

# ─── Lazy ML-service import (avoids circular import at module load) ─────────────
_svc = None

def _get_svc():
    global _svc
    if _svc is not None:
        return _svc
    try:
        from ml_service import svc as _s
        _svc = _s
        return _svc
    except Exception as exc:
        logger.warning("ml_service not available: %s", exc)
        return None

# ─── Helpers ───────────────────────────────────────────────────────────────────

def _ok(data: Any, message: str = "success") -> tuple:
    return jsonify({"status": "success", "message": message, "data": data}), 200

def _err(msg: str, code: int = 400) -> tuple:
    # Always 200 so the frontend never sees raw HTTP errors.
    # The "error" field is surfaced in the UI instead.
    return jsonify({"status": "error", "message": msg}), code

def _safe_float(val) -> Optional[float]:
    try:
        return float(val) if val not in (None, "", "None", "nan", "-") else None
    except (ValueError, TypeError):
        return None

def _clean_str(val: Any) -> str:
    """Return a clean string; treat '-', 'null', 'none', 'nan' as empty."""
    s = str(val or "").strip()
    if s.lower() in ("-", "null", "none", "nan", "n/a", "na"):
        return ""
    return s


# ─── Safety Detection ──────────────────────────────────────────────────────────

# Canonical list of harmful ingredient keywords (lowercase, substring match)
HARMFUL_INGREDIENTS: List[str] = [
    "paraben", "methylparaben", "propylparaben", "butylparaben",
    "ethylparaben", "isobutylparaben",
    "fragrance", "parfum",
    "synthetic dye", "fd&c", "d&c red", "d&c yellow",
    "talc",
    "formaldehyde", "formalin",
    "sodium lauryl sulfate", "sodium laureth sulfate", "sls", "sles",
    "phthalate", "dibutyl phthalate", "diethyl phthalate",
    "triclosan", "triclocarban",
    "hydroquinone",
    "coal tar",
    "oxybenzone", "benzophenone",
    "petrolatum", "mineral oil",
    "lead", "mercury", "arsenic",
]


def _contains_harmful(ingredients_str: str) -> bool:
    """Return True if ingredients_str contains any known harmful keyword."""
    if not ingredients_str:
        return False
    text = ingredients_str.lower()
    return any(kw in text for kw in HARMFUL_INGREDIENTS)


def _harmful_flags(ingredients_str: str) -> List[str]:
    """Return list of harmful keywords found in the ingredients string."""
    if not ingredients_str:
        return []
    text = ingredients_str.lower()
    return [kw for kw in HARMFUL_INGREDIENTS if kw in text]


# ─── Category Detection ───────────────────────────────────────────────────────

# Maps a canonical category name to a list of keywords found in product names.
# Order matters: more specific patterns should appear first.
CATEGORY_KEYWORDS: Dict[str, List[str]] = {
    "foundation":   ["foundation", "bb cream", "cc cream", "skin tint", "tinted moisturizer",
                     "tinted", "coverage", "concealer", "colour corrector", "color corrector"],
    "serum":        ["serum", "concentrate", "ampoule", "shot", "booster", "treatment essence",
                     "treatment", "activator"],
    "moisturizer":  ["moisturizer", "moisturiser", "cream", "lotion", "gel cream", "emulsion",
                     "balm", "butter", "night cream", "day cream", "hydrator", "sleeping mask",
                     "overnight mask"],
    "toner":        ["toner", "tonic", "mist", "essence", "facial water", "clarifying"],
    "cleanser":     ["cleanser", "foaming wash", "face wash", "cleansing", "makeup remover",
                     "micellar", "scrub", "exfoliant", "exfoliator", "peel"],
    "mask":         ["mask", "masque", "mud", "clay", "sheet mask", "peel-off"],
    "eye":          ["eye cream", "eye gel", "eye serum", "eye balm",
                     "under eye", "dark circle", "lash", "brow"],
    "spf":          ["spf", "sunscreen", "sun protection", "solar", "uv"],
    "lip":          ["lip", "lipstick", "lip balm", "lip gloss", "lip liner"],
    "oil":          ["facial oil", "face oil", "rosehip", "argan oil", "jojoba oil",
                     "beauty oil", "dry oil"],
    "mist":         ["mist", "spray", "facial spray", "setting spray", "rose water"],
}


def _detect_category(product_name: str) -> str:
    """
    Detect the product category from its name using keyword matching.
    Returns the category string (e.g. 'foundation') or 'general' if unknown.
    """
    name = product_name.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in name for kw in keywords):
            return category
    return "general"


# ─── CSV Readers ───────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _load_products() -> List[Dict]:
    """Load and cache de-duplicated products from master_products_cleaned.csv."""
    products: List[Dict] = []
    if not _PRODUCTS_CSV.exists():
        logger.warning("Products CSV not found: %s", _PRODUCTS_CSV)
        return products
    seen: set = set()
    try:
        with open(_PRODUCTS_CSV, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                pid = _clean_str(row.get("product_id"))
                if not pid or pid in seen:
                    continue
                seen.add(pid)
                products.append({
                    "product_id":   pid,
                    "product_name": _clean_str(row.get("product_name")),
                    "brand":        _clean_str(row.get("brand")),
                    "price":        _safe_float(row.get("price")),
                    "ingredients":  _clean_str(row.get("ingredients")),
                    "combination":  _safe_float(row.get("combination")),
                    "dry":          _safe_float(row.get("dry")),
                    "normal":       _safe_float(row.get("normal")),
                    "oily":         _safe_float(row.get("oily")),
                    "sensitive":    _safe_float(row.get("sensitive")),
                })
        logger.info("Loaded %d unique products from CSV", len(products))
    except Exception as exc:
        logger.exception("Failed to load products CSV: %s", exc)
    return products


@lru_cache(maxsize=1)
def _load_sales_index() -> Dict[str, List[float]]:
    """product_id → list of units_sold values."""
    index: Dict[str, List[float]] = {}
    if not _SALES_CSV.exists():
        return index
    try:
        with open(_SALES_CSV, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                pid   = _clean_str(row.get("product_id"))
                units = _safe_float(row.get("units_sold"))
                if pid and units is not None:
                    index.setdefault(pid, []).append(units)
    except Exception as exc:
        logger.exception("Failed to load sales CSV: %s", exc)
    return index


@lru_cache(maxsize=1)
def _load_pid_to_ingredients() -> Dict[str, str]:
    """product_id -> ingredients string from master CSV (for similarity fallback)."""
    mapping: Dict[str, str] = {}
    if not _PRODUCTS_CSV.exists():
        return mapping
    try:
        with open(_PRODUCTS_CSV, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                pid = _clean_str(row.get("product_id"))
                ing = _clean_str(row.get("ingredients"))
                if pid and ing:
                    mapping[pid] = ing
    except Exception as exc:
        logger.exception("Failed to build ingredient index: %s", exc)
    return mapping


def _load_reviews_index() -> Dict[str, str]:
    """
    product_id -> concatenated review text (up to 5 validated reviews).

    Cross-validates each review row against master_products_cleaned.csv:
    the product_name in the reviews CSV must share >=2 meaningful words with
    the known product name for that product_id.  This prevents mismatches
    caused by different source datasets reusing the same PRD_XXXXX IDs for
    completely different products.
    """
    # ── Stop-words to ignore during name comparison ───────────────────────
    _STOP = {
        "the","a","an","and","or","of","for","in","on","to","with","by","set",
        "new","mini","size","pack","kit","&","collection","limited","edition",
    }

    def _name_words(name: str) -> set:
        """Lower-case word set with stop-words removed."""
        return {w for w in name.lower().split() if w not in _STOP and len(w) > 2}

    def _names_match(name_a: str, name_b: str, threshold: int = 2) -> bool:
        """Return True if two product names share at least `threshold` words."""
        if not name_a or not name_b:
            return False
        return len(_name_words(name_a) & _name_words(name_b)) >= threshold

    # ── Build product_id -> known_name lookup from products CSV ───────────
    pid_to_known_name: Dict[str, str] = {}
    if _PRODUCTS_CSV.exists():
        try:
            with open(_PRODUCTS_CSV, newline="", encoding="utf-8", errors="replace") as f:
                for row in csv.DictReader(f):
                    pid  = _clean_str(row.get("product_id"))
                    name = _clean_str(row.get("product_name"))
                    if pid and name:
                        pid_to_known_name[pid] = name
        except Exception as exc:
            logger.warning("_load_reviews_index: could not read products CSV: %s", exc)

    # ── Build validated reviews index ─────────────────────────────────────
    if not _REVIEWS_CSV.exists():
        logger.warning("Reviews CSV not found: %s", _REVIEWS_CSV)
        return {}

    index: Dict[str, List[str]] = {}
    skipped_mismatch = 0
    skipped_blank    = 0

    try:
        with open(_REVIEWS_CSV, newline="", encoding="utf-8", errors="replace") as f:
            for row in csv.DictReader(f):
                pid        = _clean_str(row.get("product_id"))
                rev        = _clean_str(row.get("review_text"))
                review_pname = _clean_str(
                    row.get("product_name") or row.get("item_reviewed", "")
                )

                # 1. Skip rows with missing/invalid review text
                if not pid or not rev or len(rev) < 20 or rev.replace(".", "").isdigit():
                    skipped_blank += 1
                    continue

                # 2. If we know this product_id, validate the product name
                known_name = pid_to_known_name.get(pid)
                if known_name and review_pname:
                    if not _names_match(known_name, review_pname):
                        skipped_mismatch += 1
                        continue   # mismatch — different product entirely

                index.setdefault(pid, []).append(rev)

        logger.info(
            "_load_reviews_index: kept=%d pids | skipped blank=%d mismatch=%d",
            len(index), skipped_blank, skipped_mismatch,
        )
        # Collapse to up to 5 reviews per product
        return {pid: ". ".join(revs[:5]) for pid, revs in index.items()}

    except Exception as exc:
        logger.exception("Failed to build reviews index: %s", exc)
        return {}


def _product_by_id(product_id: str) -> Optional[Dict]:
    pid = product_id.strip()
    for p in _UPLOADED_PRODUCTS:
        if p.get("product_id") == pid:
            return p
    for p in _load_products():
        if p.get("product_id") == pid:
            return p
    return None


# ─── Core ingredient-based similarity ─────────────────────────────────────────

def _enrich_with_safety(results: List[Dict]) -> List[Dict]:
    """
    For each result, look up the product's ingredients and annotate:
      - is_safe: bool
      - harmful_flags: list of harmful keywords found
      - ingredients_preview: first 120 chars of ingredient string
    """
    pid_to_ing = _load_pid_to_ingredients()
    # Also check uploaded products
    uploaded_map = {p["product_id"]: p.get("ingredients", "")
                    for p in _UPLOADED_PRODUCTS}

    enriched = []
    for r in results:
        pid = r.get("product_id", "")
        ing = uploaded_map.get(pid) or pid_to_ing.get(pid, "")
        flags     = _harmful_flags(ing)
        is_safe   = len(flags) == 0
        enriched.append({
            **r,
            "is_safe":           is_safe,
            "harmful_flags":     flags,
            "ingredients_preview": ing[:120] if ing else "",
        })
    return enriched


def _ingredient_similarity(ingredients: str,
                            top_n: int,
                            exclude_name: str = "",
                            exclude_id: str = "",
                            pool_size: int = 60) -> Dict:
    """
    Compute cosine similarity between `ingredients` text and every indexed
    product using the pre-trained TF-IDF vectorizer.
    Fetches `pool_size` candidates (before safety filtering) so callers can
    always find enough safe results.
    Never raises — returns an empty results list on any error.
    """
    svc = _get_svc()
    if svc is None:
        return {"query": {}, "results": [], "top_n": top_n,
                "model_used": "unavailable",
                "error": "ML service not available"}

    tfidf = svc.tfidf
    if tfidf is None:
        return {"query": {}, "results": [], "top_n": top_n,
                "model_used": "unavailable",
                "error": "TF-IDF vectorizer not loaded"}

    if not ingredients or _clean_str(ingredients) == "":
        return {"query": {}, "results": [], "top_n": top_n,
                "model_used": "tfidf_cosine_similarity",
                "error": "No ingredients available for this product"}

    try:
        import numpy as np

        pid_to_ing    = _load_pid_to_ingredients()
        product_ids   = svc.product_ids
        product_names = svc.product_names
        corpus = [pid_to_ing.get(pid, "") for pid in product_ids]

        query_vec      = tfidf.transform([ingredients])
        product_matrix = tfidf.transform(corpus)
        scores = (query_vec * product_matrix.T).toarray().flatten()

        ranked = np.argsort(scores)[::-1]
        raw_results: List[Dict] = []
        en = exclude_name.lower().strip()
        ei = exclude_id.strip()

        for i in ranked:
            if len(raw_results) >= pool_size:
                break
            pid   = product_ids[i]   if i < len(product_ids)   else ""
            pname = product_names[i] if i < len(product_names) else ""
            sim   = float(scores[i])
            if (ei and pid == ei) or (en and pname.lower() == en):
                continue
            raw_results.append({
                "rank":          len(raw_results) + 1,
                "product_index": int(i),
                "product_id":    pid,
                "product_name":  pname,
                "similarity":    round(sim, 4),
            })

        return {
            "query": {
                "product_id":   exclude_id   or None,
                "product_name": exclude_name or None,
                "method":       "ingredient_tfidf_fallback",
            },
            "results":    raw_results,
            "top_n":      top_n,
            "model_used": "tfidf_cosine_similarity",
        }

    except Exception as exc:
        logger.exception("_ingredient_similarity error: %s", exc)
        return {"query": {}, "results": [], "top_n": top_n,
                "model_used": "tfidf_cosine_similarity",
                "error": f"Similarity computation failed: {exc}"}


# ═══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@pipeline_bp.route("/stats", methods=["GET"])
def get_stats():
    # Try to use the new persistent controller if available
    try:
        from controllers.product_intelligence_controller import get_dashboard_stats as get_new_stats
        return _ok(get_new_stats())
    except Exception as e:
        logger.warning("Falling back to legacy stats: %s", e)

    # Combine baseline data with current session's uploaded data
    products = list(_load_products()) + _UPLOADED_PRODUCTS
    
    brands = {p.get("brand") for p in products if p.get("brand")}
    with_ing = sum(1 for p in products if p.get("ingredients"))
    
    # Detect top category
    cat_counts = {}
    for p in products:
        cat = _detect_category(p.get("product_name", ""))
        if cat != "general":
            cat_counts[cat] = cat_counts.get(cat, 0) + 1
    
    top_cat = "Skincare"
    if cat_counts:
        top_cat = max(cat_counts, key=cat_counts.get).title()

    return _ok({
        "total_products": len(products),
        "total_brands":   len(brands),
        "products_with_ingredients": with_ing,
        "top_category":   top_cat,
        "data_source":    "hybrid_pipeline_store",
        "session_uploaded": len(_UPLOADED_PRODUCTS)
    })


@pipeline_bp.route("/products", methods=["GET"])
def get_products():
    page             = max(1, int(request.args.get("page",  1)))
    limit            = min(100, max(1, int(request.args.get("limit", 20))))
    search           = _clean_str(request.args.get("search", "")).lower()
    only_with_ing    = request.args.get("has_ingredients", "").lower() in ("1", "true", "yes")
    
    # Try using new persistent controller for products
    try:
        from controllers.product_intelligence_controller import get_paginated_products
        data = get_paginated_products(page, limit, search, only_with_ing)
        return _ok(data)
    except Exception as e:
        logger.warning("Falling back to legacy products: %s", e)

    only_with_review = request.args.get("has_reviews",     "").lower() in ("1", "true", "yes")
    products = list(_load_products()) + _UPLOADED_PRODUCTS

    # Annotate each product with has_reviews before filtering
    for p in products:
        p["has_reviews"] = bool(p.get("review_text", "").strip())

    if search:
        products = [p for p in products
                    if search in p["product_name"].lower() or search in (p.get("brand") or "").lower()]
    if only_with_ing:
        products = [p for p in products if p.get("ingredients")]
    if only_with_review:
        products = [p for p in products if p["has_reviews"]]

    total = len(products)
    start = (page - 1) * limit
    return _ok({
        "products":    products[start : start + limit],
        "total":       total,
        "page":        page,
        "limit":       limit,
        "total_pages": max(1, (total + limit - 1) // limit),
    })


@pipeline_bp.route("/product/<product_id>", methods=["GET"])
def get_product(product_id: str):
    p = _product_by_id(product_id)
    if not p:
        return _err(f"Product '{product_id}' not found.", 404)
    sales_index   = _load_sales_index()
    sales_history = sales_index.get(product_id, [])
    recent_sales  = sales_history[-12:] if len(sales_history) >= 12 else sales_history
    result = dict(p)
    result["sales_history"] = sales_history
    result["recent_sales"]  = recent_sales
    result["sales_count"]   = len(sales_history)
    # review_text comes only from the uploaded product dict — no cross-CSV lookup
    result["has_reviews"]   = bool(result.get("review_text", "").strip())
    return _ok(result)


@pipeline_bp.route("/upload", methods=["POST"])
def upload_csv():
    """Multi-tenant CSV upload handler."""
    try:
        from controllers.product_intelligence_controller import handle_product_upload
        # handle_product_upload now handles its own auth internally via token check
        return handle_product_upload(request)
    except Exception as e:
        logger.error("Upload route failed: %s", e)
        return jsonify({"status": "error", "message": str(e)}), 500
@pipeline_bp.route("/report/<product_id>", methods=["GET"])
def get_report(product_id: str):
    p = _product_by_id(product_id) or {
        "product_id": product_id, "product_name": "Unknown", "brand": "—"
    }
    sales_index   = _load_sales_index()
    sales_history = sales_index.get(product_id, [])
    return _ok({
        "product":       p,
        "sales_history": sales_history[-12:],
        "sales_total":   sum(sales_history),
        "report_generated_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "pipeline_version": "2.0",
    })


# ─── Unified smart similarity endpoint ────────────────────────────────────────

@pipeline_bp.route("/smart-similarity", methods=["POST"])
def smart_similarity():
    """
    Unified similarity endpoint that ALWAYS returns results:

    Strategy (in priority order):
      1. If product_id or product_name is in the pre-built ML index → use index.
      2. If not in index AND ingredients are provided → ingredient TF-IDF fallback.
      3. If not in index AND no ingredients → return clear error (no 404).

    Body (JSON):
      {
        "product_name": "GlowCo Advanced Lotion",    // optional
        "product_id":   "PRD_008",                   // optional
        "ingredients":  "Water, Glycerin, ...",       // optional but needed for fallback
        "top_n":        6
      }

    Returns the SAME shape as /api/similar-products.
    HTTP status is always 200 — errors are in the response body.
    """
    body         = request.get_json(force=True, silent=True) or {}
    product_name = _clean_str(body.get("product_name"))
    product_id   = _clean_str(body.get("product_id"))
    ingredients  = _clean_str(body.get("ingredients"))
    top_n        = max(1, min(20, int(body.get("top_n", 6))))

    console_info = f"smart-similarity | pid={product_id!r} name={product_name!r} ing_len={len(ingredients)}"
    logger.info(console_info)

    svc = _get_svc()

    # Detect whether the query product itself is unsafe
    is_unsafe = _contains_harmful(ingredients)
    logger.info("smart-similarity: is_unsafe=%s ing_len=%d", is_unsafe, len(ingredients))

    # ── Attempt 1: Standard ML index lookup ───────────────────────────────────
    raw_results: List[Dict] = []
    method = "index"

    if svc is not None and (product_id or product_name):
        idx = svc._resolve_product_index(
            None,
            product_id   or None,
            product_name or None,
        )
        if idx is not None:
            # Fetch a larger pool so after filtering we still have enough
            pool = svc.get_similar_products(product_index=idx, top_n=top_n * 15)
            raw_results = pool.get("results", [])
            logger.info("smart-similarity: index hit idx=%d pool=%d", idx, len(raw_results))
        else:
            logger.info("smart-similarity: not in index, trying ingredient fallback")
            method = "ingredient_tfidf"

    # ── Attempt 2: TF-IDF ingredient fallback ─────────────────────────────────
    if not raw_results:
        method = "ingredient_tfidf"
        # Auto-resolve ingredients if not provided in body
        if not ingredients and product_id:
            for p in _UPLOADED_PRODUCTS:
                if p.get("product_id") == product_id:
                    ingredients = p.get("ingredients", "")
                    break
            if not ingredients:
                p2 = _product_by_id(product_id)
                if p2:
                    ingredients = p2.get("ingredients", "")

        if not ingredients and product_name:
            for p in list(_load_products()) + _UPLOADED_PRODUCTS:
                if p.get("product_name", "").lower() == product_name.lower():
                    ingredients = p.get("ingredients", "")
                    break

        if not ingredients:
            logger.info("smart-similarity: no ingredients available")
            return _ok({
                "query":            {"product_id": product_id, "product_name": product_name},
                "results":          [],
                "top_n":            top_n,
                "model_used":       "none",
                "mode":             "no_ingredients",
                "original_is_unsafe": is_unsafe,
                "_method":          "no_ingredients",
                "error":            "No ingredients available for this product. "
                                    "Go to Ingredient Safety first to load the product.",
            })

        sim_result  = _ingredient_similarity(
            ingredients, top_n,
            exclude_name=product_name,
            exclude_id=product_id,
            pool_size=top_n * 15,
        )
        raw_results = sim_result.get("results", [])
        is_unsafe   = is_unsafe or _contains_harmful(ingredients)

    # ── Detect query product category ─────────────────────────────────────────
    query_category = _detect_category(product_name or "")
    logger.info("smart-similarity: query_category=%r", query_category)

    # ── Enrich results with safety + category data ────────────────────────────
    enriched = _enrich_with_safety(raw_results)
    # Annotate each result with its detected category
    for r in enriched:
        r["category"] = _detect_category(r.get("product_name", ""))

    # ── Step 1: Always keep only SAFE products ────────────────────────────────
    safe_pool = [r for r in enriched if r["is_safe"]]
    logger.info("smart-similarity: safe_pool=%d / total=%d", len(safe_pool), len(enriched))

    # ── Step 2: Filter to the same category ───────────────────────────────────
    if query_category != "general":
        same_cat = [r for r in safe_pool if r["category"] == query_category]
        logger.info("smart-similarity: same_cat_safe=%d (cat=%r)", len(same_cat), query_category)
    else:
        same_cat = safe_pool  # unknown category — return all safe

    # ── Step 3: Fallback logic ────────────────────────────────────────────────
    #   If we couldn't find enough same-category safe products, gracefully fall
    #   back to (a) all safe products, or (b) empty with a clear message.
    if len(same_cat) >= 1:
        final = same_cat[:top_n]
        mode  = "safe_filtered"
        category_filtered = (query_category != "general")
    elif safe_pool:
        # There are safe products but none match the category — return safe products
        # with a note that category filtering was relaxed.
        final = safe_pool[:top_n]
        mode  = "safe_filtered"
        category_filtered = False
        logger.info("smart-similarity: no same-category safe results, relaxing category filter")
    else:
        final = []
        mode  = "no_safe_alternatives"
        category_filtered = False

    # Re-number ranks after filtering
    for i, r in enumerate(final):
        r["rank"] = i + 1

    # ── Handle no safe alternatives ────────────────────────────────────────────
    if not final:
        return _ok({
            "query":              {"product_id": product_id, "product_name": product_name},
            "results":            [],
            "top_n":              top_n,
            "model_used":         "tfidf_cosine_similarity",
            "mode":               "no_safe_alternatives",
            "original_is_unsafe": is_unsafe,
            "query_category":     query_category,
            "_method":            method,
            "message":            "No safe alternatives found in the same category. "
                                  "All similar products contain harmful ingredients.",
        })

    logger.info("smart-similarity: final=%d mode=%s cat_filtered=%s",
                len(final), mode, category_filtered)
    return _ok({
        "query":              {"product_id": product_id, "product_name": product_name, "method": method},
        "results":            final,
        "top_n":              top_n,
        "model_used":         "tfidf_cosine_similarity",
        "mode":               mode,
        "original_is_unsafe": is_unsafe,
        "query_category":     query_category,
        "category_filtered":  category_filtered,
        "_method":            method,
    })


# ─── Explicit ingredient-based endpoint (called directly from frontend) ────────

@pipeline_bp.route("/similar-by-ingredients", methods=["POST"])
def similar_by_ingredients():
    """
    Ingredient-text similarity search for any product.
    Body: { ingredients, product_name, product_id, top_n }
    """
    body         = request.get_json(force=True, silent=True) or {}
    ingredients  = _clean_str(body.get("ingredients"))
    product_name = _clean_str(body.get("product_name"))
    product_id   = _clean_str(body.get("product_id"))
    top_n        = max(1, min(20, int(body.get("top_n", 6))))
    filter_safe  = bool(body.get("filter_safe", False))

    if not ingredients:
        return _ok({
            "query":      {"product_id": product_id, "product_name": product_name},
            "results":    [],
            "top_n":      top_n,
            "model_used": "none",
            "mode":       "no_ingredients",
            "original_is_unsafe": False,
            "error":      "No ingredients available for this product.",
        })

    is_unsafe = _contains_harmful(ingredients)
    sim = _ingredient_similarity(
        ingredients, top_n,
        exclude_name=product_name,
        exclude_id=product_id,
        pool_size=top_n * 15,
    )
    enriched = _enrich_with_safety(sim.get("results", []))

    if filter_safe or is_unsafe:
        final = [r for r in enriched if r["is_safe"]][:top_n]
        mode  = "safe_filtered"
    else:
        final = enriched[:top_n]
        mode  = "normal"

    for i, r in enumerate(final):
        r["rank"] = i + 1

    sim["results"]           = final
    sim["mode"]              = mode
    sim["original_is_unsafe"] = is_unsafe
    return _ok(sim)
