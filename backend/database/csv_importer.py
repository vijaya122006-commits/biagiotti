"""
database/csv_importer.py — One-time import of all CSV datasets into SQLite/PostgreSQL.
Run via: python backend/setup.py
"""
from __future__ import annotations
import hashlib
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from database.models import db, Product, Sale, Review, HarmfulChemical, Inventory

logger = logging.getLogger("csv_importer")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

_DATA = _BACKEND / "data"
SEED_DEALER_ID = 1


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _brand_id(brand: str) -> str:
    """Generate a deterministic brand ID from the brand name."""
    h = int(hashlib.md5(str(brand).encode()).hexdigest(), 16) % 100000
    return f"BRD_{h:05d}"


def _skin_suitability(row: pd.Series, cols=None) -> str:
    """Convert boolean skin-type columns to a comma-separated suitability string."""
    if cols is None:
        cols = ['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive']
    types = []
    name_map = {
        'Combination': 'combination', 'Dry': 'dry', 'Normal': 'normal',
        'Oily': 'oily', 'Sensitive': 'sensitive',
        'combination': 'combination', 'dry': 'dry', 'normal': 'normal',
        'oily': 'oily', 'sensitive': 'sensitive',
    }
    for c in cols:
        if c in row.index:
            try:
                val = row[c]
                if pd.notna(val) and str(val) not in ('0', '0.0', 'False', 'false', ''):
                    types.append(name_map.get(c, c.lower()))
            except Exception:
                pass
    return ', '.join(types) if types else 'all'


def _safe_float(val, default=None):
    try:
        return float(val) if pd.notna(val) else default
    except Exception:
        return default


def _safe_str(val, default='') -> str:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return default
    return str(val).strip()


def _exists_product(product_id: str, dealer_id: int) -> bool:
    return db.session.query(
        Product.query.filter_by(product_id=product_id, dealer_id=dealer_id).exists()
    ).scalar()


def _bulk_commit(items: list, label: str):
    """Add list of model objects and commit, logging progress."""
    if not items:
        logger.info("  No new rows to import for %s", label)
        return
    try:
        db.session.bulk_save_objects(items)
        db.session.commit()
        logger.info("  Imported %d rows from %s", len(items), label)
    except Exception as e:
        db.session.rollback()
        logger.error("  Failed to import %s: %s", label, e)


# ─── File 1: cosmetic_p.csv ────────────────────────────────────────────────────

def import_cosmetic_p():
    path = _DATA / "products folder" / "cosmetic_p.csv"
    if not path.exists():
        logger.warning("SKIP — %s not found", path)
        return

    df = pd.read_csv(path, on_bad_lines='skip', low_memory=False)
    logger.info("Loading %s (%d rows)...", path.name, len(df))

    skin_cols = [c for c in ['Combination', 'Dry', 'Normal', 'Oily', 'Sensitive'] if c in df.columns]
    items = []
    seen = set()
    counter = 1

    for _, row in df.iterrows():
        pname = _safe_str(row.get('name') or row.get('product_name'))
        if not pname:
            continue

        pid = f"PRD_{counter:05d}"
        counter += 1

        key = (pid, SEED_DEALER_ID)
        if key in seen:
            continue
        seen.add(key)

        brand = _safe_str(row.get('brand'))
        skin = _skin_suitability(row, skin_cols)
        price = _safe_float(row.get('price'))

        items.append(Product(
            product_id=pid,
            dealer_id=SEED_DEALER_ID,
            product_name=pname,
            brand=brand,
            brand_id=_brand_id(brand) if brand else None,
            category=_safe_str(row.get('Label') or row.get('label')),
            price=price,
            cost_price=round(price * 0.4, 2) if price else None,
            ingredients=_safe_str(row.get('ingredients')),
            skin_suitability=skin,
            is_verified=False,
        ))

        if len(items) >= 500:
            _bulk_commit(items, "cosmetic_p.csv (batch)")
            items = []

    _bulk_commit(items, "cosmetic_p.csv")


# ─── File 2: master_products_cleaned.csv ──────────────────────────────────────

def import_master_products():
    path = _DATA / "cleaned" / "master_products_cleaned.csv"
    if not path.exists():
        logger.warning("SKIP — %s not found", path)
        return

    df = pd.read_csv(path, on_bad_lines='skip', low_memory=False)
    logger.info("Loading %s (%d rows)...", path.name, len(df))

    skin_cols = [c for c in ['combination', 'dry', 'normal', 'oily', 'sensitive'] if c in df.columns]
    seen_pids: set = set()
    items = []

    for _, row in df.iterrows():
        pid = _safe_str(row.get('product_id'))
        pname = _safe_str(row.get('product_name'))
        if not pid and not pname:
            continue
        if not pid:
            pid = f"MSTR_{abs(hash(pname)) % 100000:05d}"

        if pid in seen_pids:
            continue
        seen_pids.add(pid)

        # Skip if already in DB
        if Product.query.filter_by(product_id=pid, dealer_id=SEED_DEALER_ID).first():
            continue

        brand = _safe_str(row.get('brand'))
        price = _safe_float(row.get('price'))
        skin = _skin_suitability(row, skin_cols)

        items.append(Product(
            product_id=pid,
            dealer_id=SEED_DEALER_ID,
            product_name=pname,
            brand=brand,
            brand_id=_safe_str(row.get('brand_id')) or _brand_id(brand),
            category=_safe_str(row.get('label') or row.get('category')),
            price=price,
            cost_price=round(price * 0.4, 2) if price else None,
            ingredients=_safe_str(row.get('ingredients')),
            skin_suitability=skin,
            monk_category=_safe_float(row.get('monk_category')),
            shade=_safe_str(row.get('shade_name')),
            hex_color=_safe_str(row.get('hex')),
            is_verified=False,
        ))

        if len(items) >= 500:
            _bulk_commit(items, "master_products_cleaned.csv (batch)")
            items = []

    _bulk_commit(items, "master_products_cleaned.csv")


# ─── File 3: master_sales_cleaned.csv ─────────────────────────────────────────

def import_master_sales():
    path = _DATA / "cleaned" / "master_sales_cleaned.csv"
    if not path.exists():
        logger.warning("SKIP — %s not found", path)
        return

    df = pd.read_csv(path, on_bad_lines='skip', low_memory=False)
    logger.info("Loading %s (%d rows)...", path.name, len(df))

    items = []
    for _, row in df.iterrows():
        pid = _safe_str(row.get('product_id') or row.get('sku'))
        if not pid:
            continue

        # Parse date
        sale_date = None
        year = _safe_float(row.get('year'))
        month = _safe_float(row.get('month'))

        raw_date = row.get('start_date') or row.get('sale_date')
        if raw_date and pd.notna(raw_date):
            try:
                sale_date = pd.to_datetime(raw_date, errors='coerce')
                if pd.notna(sale_date):
                    year = year or sale_date.year
                    month = month or sale_date.month
                    sale_date = sale_date.to_pydatetime()
                else:
                    sale_date = None
            except Exception:
                sale_date = None

        price = _safe_float(row.get('price') or row.get('price_usd'), 0)
        units = _safe_float(row.get('units_sold'), 0)

        items.append(Sale(
            product_id=pid,
            dealer_id=SEED_DEALER_ID,
            brand=_safe_str(row.get('brand')),
            region=_safe_str(row.get('region')),
            city=_safe_str(row.get('city')),
            event_type=_safe_str(row.get('event_type')),
            year=int(year) if year else None,
            month=int(month) if month else None,
            units_sold=units,
            revenue=round(price * units, 2) if price and units else 0,
            sell_through_pct=_safe_float(row.get('sell_through_pct')),
            avg_daily_footfall=_safe_float(row.get('avg_daily_footfall')),
            sale_date=sale_date,
        ))

        if len(items) >= 1000:
            _bulk_commit(items, "master_sales_cleaned.csv (batch)")
            items = []

    _bulk_commit(items, "master_sales_cleaned.csv")


# ─── File 4: Chemicals in Makeup.csv ──────────────────────────────────────────

def import_harmful_chemicals():
    path = _DATA / "ingredients folder" / "Chemicals in Makeup.csv"
    if not path.exists():
        logger.warning("SKIP — %s not found", path)
        return

    # Check if already imported
    if HarmfulChemical.query.count() > 0:
        logger.info("SKIP — harmful chemicals already imported")
        return

    df = pd.read_csv(path, on_bad_lines='skip', low_memory=False)
    logger.info("Loading %s (%d rows)...", path.name, len(df))

    seen_names: set = set()
    items = []

    for _, row in df.iterrows():
        cname = _safe_str(row.get('ChemicalName'))
        if not cname or cname in seen_names:
            continue
        seen_names.add(cname)

        lower = cname.lower()
        if 'mercury' in lower or 'lead' in lower:
            severity = 10
        elif 'formaldehyde' in lower:
            severity = 9
        elif 'phthalate' in lower or 'coal tar' in lower:
            severity = 8
        elif 'paraben' in lower:
            severity = 7
        else:
            severity = 5

        risk = 'high' if severity >= 8 else ('medium' if severity >= 5 else 'low')

        items.append(HarmfulChemical(
            chemical_name=cname[:300],
            cas_number=_safe_str(row.get('CasNumber')),
            risk_level=risk,
            health_risk=f"Potential health concern — severity {severity}/10",
            legal_status='Reported to CDPH',
            primary_category=_safe_str(row.get('PrimaryCategory')),
            sub_category=_safe_str(row.get('SubCategory')),
            severity_score=severity,
            source_dataset='Chemicals in Makeup.csv',
        ))

        if len(items) >= 500:
            _bulk_commit(items, "Chemicals in Makeup.csv (batch)")
            items = []

    _bulk_commit(items, "Chemicals in Makeup.csv")


# ─── File 5: master_reviews_cleaned.csv ───────────────────────────────────────

def import_master_reviews():
    path = _DATA / "cleaned" / "master_reviews_cleaned.csv"
    if not path.exists():
        logger.warning("SKIP — %s not found", path)
        return

    df = pd.read_csv(path, nrows=5000, on_bad_lines='skip', low_memory=False)
    logger.info("Loading %s (%d rows, capped at 5000)...", path.name, len(df))

    items = []
    for _, row in df.iterrows():
        pid = _safe_str(row.get('product_id'))
        if not pid:
            continue

        rating = _safe_float(row.get('rating') or row.get('aggregate_rating'))
        review_text = _safe_str(row.get('review_text') or row.get('description'))
        if not review_text:
            continue

        # Parse skin type from afteruse field
        afteruse = _safe_str(row.get('afteruse', '')).lower()
        skin_type = ''
        for st in ['oily', 'dry', 'sensitive', 'normal', 'combination', 'acne']:
            if st in afteruse:
                skin_type = st
                break

        items.append(Review(
            product_id=pid,
            dealer_id=SEED_DEALER_ID,
            source='sephora',
            platform='sephora',
            rating=rating,
            review_body=review_text[:2000],
            skin_type_mentioned=skin_type,
            verified_purchase=False,
            is_synthetic=False,
        ))

        if len(items) >= 500:
            _bulk_commit(items, "master_reviews_cleaned.csv (batch)")
            items = []

    _bulk_commit(items, "master_reviews_cleaned.csv")


# ─── File 6: review_data.csv ──────────────────────────────────────────────────

def import_review_data():
    path = _DATA / "reviews folder" / "review_data.csv"
    if not path.exists():
        logger.warning("SKIP — %s not found", path)
        return

    df = pd.read_csv(path, nrows=3000, on_bad_lines='skip', low_memory=False)
    logger.info("Loading %s (%d rows, capped at 3000)...", path.name, len(df))

    items = []
    for _, row in df.iterrows():
        pid = _safe_str(row.get('product_id') or row.get('item_reviewed'))
        if not pid:
            pid = f"REV_{abs(hash(_safe_str(row.get('item_reviewed', '')))) % 100000:05d}"

        rating = _safe_float(row.get('rating_value'))
        body = _safe_str(row.get('text'))
        if not body:
            continue

        review_date = None
        raw = row.get('date_published')
        if raw and pd.notna(raw):
            try:
                dt = pd.to_datetime(raw, errors='coerce')
                if pd.notna(dt):
                    review_date = dt.to_pydatetime()
            except Exception:
                pass

        label = _safe_str(row.get('label', '')).lower()
        skin_type = ''
        for st in ['oily', 'dry', 'sensitive', 'normal', 'combination']:
            if st in label:
                skin_type = st
                break

        items.append(Review(
            product_id=pid,
            dealer_id=SEED_DEALER_ID,
            source='sephora_external',
            platform='sephora',
            rating=rating,
            review_title=_safe_str(row.get('title'))[:500] if row.get('title') else None,
            review_body=body[:2000],
            skin_type_mentioned=skin_type,
            is_synthetic=False,
            review_date=review_date,
        ))

        if len(items) >= 500:
            _bulk_commit(items, "review_data.csv (batch)")
            items = []

    _bulk_commit(items, "review_data.csv")


# ─── File 7: verified_products.csv ────────────────────────────────────────────

def import_verified_products():
    path = _DATA / "verified_products" / "verified_products.csv"
    if not path.exists():
        logger.warning("SKIP — %s not found", path)
        return

    df = pd.read_csv(path, on_bad_lines='skip', low_memory=False)
    logger.info("Loading %s (%d rows)...", path.name, len(df))

    items = []
    for _, row in df.iterrows():
        pid = _safe_str(row.get('product_id'))
        pname = _safe_str(row.get('product_name'))
        if not pid and not pname:
            continue
        if not pid:
            pid = f"VRF_{abs(hash(pname)) % 100000:05d}"

        if Product.query.filter_by(product_id=pid, dealer_id=SEED_DEALER_ID).first():
            continue

        price = _safe_float(row.get('price'))
        brand = _safe_str(row.get('brand'))

        items.append(Product(
            product_id=pid,
            dealer_id=SEED_DEALER_ID,
            product_name=pname,
            brand=brand,
            brand_id=_brand_id(brand) if brand else None,
            price=price,
            cost_price=round(price * 0.4, 2) if price else None,
            ingredients=_safe_str(row.get('ingredients')),
            skin_suitability=_safe_str(row.get('skin_suitability')),
            shade=_safe_str(row.get('shade')),
            monk_category=_safe_float(row.get('monk_category')),
            is_verified=bool(row.get('verified_flag', 0)),
        ))

        if len(items) >= 500:
            _bulk_commit(items, "verified_products.csv (batch)")
            items = []

    _bulk_commit(items, "verified_products.csv")


# ─── File 8: luxury_cosmetics_popups.csv ──────────────────────────────────────

def import_luxury_sales():
    path = _DATA / "sales folder" / "luxury_cosmetics_popups.csv"
    if not path.exists():
        logger.warning("SKIP — %s not found", path)
        return

    df = pd.read_csv(path, on_bad_lines='skip', low_memory=False)
    logger.info("Loading %s (%d rows)...", path.name, len(df))

    items = []
    for _, row in df.iterrows():
        pid = _safe_str(row.get('product_id') or row.get('sku'))
        if not pid:
            continue

        price = _safe_float(row.get('price_usd') or row.get('price'), 0)
        units = _safe_float(row.get('units_sold'), 0)

        sale_date = None
        year = None
        month = None
        raw_date = row.get('start_date')
        if raw_date and pd.notna(raw_date):
            try:
                dt = pd.to_datetime(raw_date, errors='coerce')
                if pd.notna(dt):
                    year = dt.year
                    month = dt.month
                    sale_date = dt.to_pydatetime()
            except Exception:
                pass

        items.append(Sale(
            product_id=pid,
            dealer_id=SEED_DEALER_ID,
            brand=_safe_str(row.get('brand')),
            region=_safe_str(row.get('region')),
            city=_safe_str(row.get('city')),
            event_type=_safe_str(row.get('event_type')),
            year=year,
            month=month,
            units_sold=units,
            revenue=round(price * units, 2) if units else 0,
            sell_through_pct=_safe_float(row.get('sell_through_pct')),
            avg_daily_footfall=_safe_float(row.get('avg_daily_footfall')),
            sale_date=sale_date,
        ))

        if len(items) >= 500:
            _bulk_commit(items, "luxury_cosmetics_popups.csv (batch)")
            items = []

    _bulk_commit(items, "luxury_cosmetics_popups.csv")


# ─── File 9: datasheet.csv ────────────────────────────────────────────────────

def import_datasheet():
    path = _DATA / "ingredients folder" / "datasheet.csv"
    if not path.exists():
        logger.warning("SKIP — %s not found", path)
        return

    df = pd.read_csv(path, nrows=3000, on_bad_lines='skip', low_memory=False)
    logger.info("Loading %s (%d rows, capped at 3000)...", path.name, len(df))

    items = []
    for _, row in df.iterrows():
        pname = _safe_str(row.get('name') or row.get('product_name'))
        if not pname:
            continue

        pid = f"DS_{abs(hash(pname)) % 100000:05d}"

        if Product.query.filter_by(product_id=pid, dealer_id=SEED_DEALER_ID).first():
            continue

        # Parse skin suitability from afterUse
        afteruse = _safe_str(row.get('afterUse') or row.get('afteruse', '')).lower()
        skin_type = 'all'
        for st in ['oily', 'dry', 'sensitive', 'normal', 'combination']:
            if st in afteruse:
                skin_type = st
                break

        brand = _safe_str(row.get('brand'))
        items.append(Product(
            product_id=pid,
            dealer_id=SEED_DEALER_ID,
            product_name=pname,
            brand=brand,
            brand_id=_brand_id(brand) if brand else None,
            category=_safe_str(row.get('type') or row.get('label')),
            ingredients=_safe_str(row.get('ingridients') or row.get('ingredients')),
            skin_suitability=skin_type,
            country=_safe_str(row.get('country')),
            is_verified=False,
        ))

        if len(items) >= 500:
            _bulk_commit(items, "datasheet.csv (batch)")
            items = []

    _bulk_commit(items, "datasheet.csv")


# ─── Inventory seeding ────────────────────────────────────────────────────────

def seed_inventory_for_imported_products():
    """Create inventory records for all products that don't have one yet."""
    import random
    products = Product.query.filter_by(dealer_id=SEED_DEALER_ID).all()
    items = []
    for p in products:
        if Inventory.query.filter_by(product_id=p.product_id, dealer_id=SEED_DEALER_ID).first():
            continue
        stock = random.randint(50, 500)
        items.append(Inventory(
            product_id=p.product_id,
            dealer_id=SEED_DEALER_ID,
            current_stock=stock,
            reorder_level=50,
            lead_time_days=14,
            last_restocked=datetime.utcnow(),
        ))

        if len(items) >= 500:
            _bulk_commit(items, "inventory seed (batch)")
            items = []

    _bulk_commit(items, "inventory seed")


# ─── Master entry point ────────────────────────────────────────────────────────

def import_all_csvs():
    """Import all 9 CSV files into the database. Idempotent — skips duplicates."""
    logger.info("=" * 50)
    logger.info("Starting CSV import pipeline...")
    logger.info("=" * 50)

    import_harmful_chemicals()
    import_cosmetic_p()
    import_master_products()
    import_master_sales()
    import_verified_products()
    import_luxury_sales()
    import_master_reviews()
    import_review_data()
    import_datasheet()
    seed_inventory_for_imported_products()

    total_products = Product.query.filter_by(dealer_id=SEED_DEALER_ID).count()
    total_sales = Sale.query.filter_by(dealer_id=SEED_DEALER_ID).count()
    total_reviews = Review.query.filter_by(dealer_id=SEED_DEALER_ID).count()
    total_chemicals = HarmfulChemical.query.count()

    logger.info("=" * 50)
    logger.info("Import complete!")
    logger.info("  Products  : %d", total_products)
    logger.info("  Sales     : %d", total_sales)
    logger.info("  Reviews   : %d", total_reviews)
    logger.info("  Chemicals : %d", total_chemicals)
    logger.info("=" * 50)
