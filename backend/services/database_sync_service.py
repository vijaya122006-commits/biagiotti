"""
services/database_sync_service.py
──────────────────────────────────
Connects to a dealer's own external database (PostgreSQL / MySQL) and
pulls their products + sales data into the Biagiotti multi-tenant schema.

Flow:
  1. Dealer saves their DB credentials via POST /api/db-connect/save
  2. We test the connection immediately
  3. A background thread runs sync_dealer_database() every N minutes
  4. After each sync, POST /api/pipeline/refresh is triggered automatically
"""
from __future__ import annotations

import base64
import json
import logging
import threading
import time
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger("db_sync")

# ── Password helpers (simple base64 — swap for vault/KMS in production) ────────

def _encode_password(plain: str) -> str:
    return base64.b64encode(plain.encode()).decode()

def _decode_password(encoded: str) -> str:
    try:
        return base64.b64decode(encoded.encode()).decode()
    except Exception:
        return ""


# ── Build a SQLAlchemy engine for the dealer's external database ────────────────

def _build_engine(conn):
    """Create a SQLAlchemy engine from a DealerDatabaseConnection row."""
    from sqlalchemy import create_engine

    db_type = (conn.db_type or "").lower()

    if db_type == "sqlite":
        # db_name holds the absolute file path for local SQLite testing
        url = f"sqlite:///{conn.db_name}"
        return create_engine(url, connect_args={"check_same_thread": False})

    password = _decode_password(conn.password_encrypted or "")
    if db_type == "postgresql":
        driver = "postgresql+psycopg2"
    elif db_type == "mysql":
        driver = "mysql+pymysql"
    else:
        raise ValueError(f"Unsupported db_type: {db_type!r}. Use 'postgresql', 'mysql', or 'sqlite'.")

    url = f"{driver}://{conn.username}:{password}@{conn.host}:{conn.port}/{conn.db_name}"
    return create_engine(url, connect_args={"connect_timeout": 10}, pool_pre_ping=True)


# ── Test a connection without persisting anything ───────────────────────────────

def test_connection(db_type: str, host: str, port: int, db_name: str,
                    username: str, password: str) -> Dict[str, Any]:
    """
    Try connecting to the dealer's external DB.
    Returns {"success": True/False, "message": str, "tables": [...]}
    Supports: postgresql, mysql, sqlite (db_name = absolute file path).
    """
    from sqlalchemy import create_engine, text, inspect
    import os

    db_type = db_type.lower()

    # ── SQLite: file-path based, no server needed ──────────────────────────────
    if db_type == "sqlite":
        if not os.path.isfile(db_name):
            return {"success": False, "message": f"SQLite file not found: {db_name}", "tables": []}
        url = f"sqlite:///{db_name}"
        try:
            engine = create_engine(url, connect_args={"check_same_thread": False})
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            inspector = inspect(engine)
            tables = inspector.get_table_names()
            engine.dispose()
            return {"success": True, "message": "SQLite connection successful!", "tables": tables}
        except Exception as e:
            return {"success": False, "message": str(e), "tables": []}

    # ── PostgreSQL / MySQL ─────────────────────────────────────────────────────
    if db_type == "postgresql":
        driver = "postgresql+psycopg2"
    elif db_type == "mysql":
        driver = "mysql+pymysql"
    else:
        return {"success": False, "message": f"Unsupported type: {db_type}. Use postgresql, mysql, or sqlite.", "tables": []}

    url = f"{driver}://{username}:{password}@{host}:{port}/{db_name}"
    try:
        engine = create_engine(url, connect_args={"connect_timeout": 8}, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        engine.dispose()
        return {"success": True, "message": "Connection successful!", "tables": tables}
    except Exception as e:
        return {"success": False, "message": str(e), "tables": []}


# ── Column mapping helpers ──────────────────────────────────────────────────────

_DEFAULT_PRODUCT_MAP = {
    # biagiotti_column : dealer_column (dealer may rename these)
    "product_id":       "product_id",
    "product_name":     "product_name",
    "brand":            "brand",
    "category":         "category",
    "price":            "price",
    "ingredients":      "ingredients",
    "skin_suitability": "skin_suitability",
}

_DEFAULT_SALES_MAP = {
    "product_id":  "product_id",
    "units_sold":  "units_sold",
    "revenue":     "revenue",
    "year":        "year",
    "month":       "month",
    "region":      "region",
    "brand":       "brand",
}


def _resolve_col(row_dict: dict, col_map: dict, biagiotti_col: str, default=None):
    """Look up a value from the dealer's row using column mapping."""
    dealer_col = col_map.get(biagiotti_col, biagiotti_col)
    return row_dict.get(dealer_col, default)


# ── Core sync function ──────────────────────────────────────────────────────────

def sync_dealer_database(dealer_id: int) -> Dict[str, Any]:
    """
    Pull products + sales from the dealer's external DB into our schema.
    Called by the background sync thread and by the manual-sync endpoint.
    """
    from database.models import db, DealerDatabaseConnection, Product, Sale

    conn_row = DealerDatabaseConnection.query.filter_by(dealer_id=dealer_id, is_active=True).first()
    if not conn_row:
        return {"success": False, "message": "No active database connection found for this dealer."}

    # Mark as running
    conn_row.sync_status = "running"
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()

    try:
        engine = _build_engine(conn_row)
        col_map = json.loads(conn_row.column_map_json or "{}") or {}
        product_map = {**_DEFAULT_PRODUCT_MAP, **col_map.get("products", {})}
        sales_map   = {**_DEFAULT_SALES_MAP,   **col_map.get("sales", {})}

        total_rows = 0

        # ── Sync Products ──────────────────────────────────────────────────────
        products_table = conn_row.products_table or "products"
        try:
            from sqlalchemy import text as _text
            with engine.connect() as ext_conn:
                result = ext_conn.execute(_text(f"SELECT * FROM {products_table} LIMIT 50000"))
                rows = [dict(r._mapping) for r in result]

            for row in rows:
                try:
                    pid = str(_resolve_col(row, product_map, "product_id", ""))
                    if not pid:
                        continue
                    existing = Product.query.filter_by(product_id=pid, dealer_id=dealer_id).first()
                    if existing:
                        existing.product_name    = str(_resolve_col(row, product_map, "product_name", existing.product_name))
                        existing.brand           = str(_resolve_col(row, product_map, "brand", existing.brand or ""))
                        existing.category        = str(_resolve_col(row, product_map, "category", existing.category or ""))
                        existing.ingredients     = str(_resolve_col(row, product_map, "ingredients", existing.ingredients or ""))
                        _price = _resolve_col(row, product_map, "price", None)
                        if _price is not None:
                            existing.price = float(_price)
                    else:
                        _price = _resolve_col(row, product_map, "price", None)
                        product = Product(
                            product_id       = pid,
                            dealer_id        = dealer_id,
                            product_name     = str(_resolve_col(row, product_map, "product_name", pid)),
                            brand            = str(_resolve_col(row, product_map, "brand", "")),
                            category         = str(_resolve_col(row, product_map, "category", "")),
                            price            = float(_price) if _price is not None else None,
                            ingredients      = str(_resolve_col(row, product_map, "ingredients", "")),
                            skin_suitability = str(_resolve_col(row, product_map, "skin_suitability", "all")),
                        )
                        db.session.add(product)
                    total_rows += 1
                except Exception as row_err:
                    logger.warning("Product row skip: %s", row_err)
                    continue

            db.session.commit()
        except Exception as prod_err:
            db.session.rollback()
            logger.error("Products table sync failed: %s", prod_err)

        # ── Sync Sales ─────────────────────────────────────────────────────────
        sales_table = conn_row.sales_table or "sales"
        try:
            now = datetime.utcnow()
            with engine.connect() as ext_conn:
                result = ext_conn.execute(_text(f"SELECT * FROM {sales_table} LIMIT 200000"))
                rows = [dict(r._mapping) for r in result]

            for row in rows:
                try:
                    pid = str(_resolve_col(row, sales_map, "product_id", ""))
                    if not pid:
                        continue
                    sale = Sale(
                        product_id = pid,
                        dealer_id  = dealer_id,
                        units_sold = float(_resolve_col(row, sales_map, "units_sold", 0) or 0),
                        revenue    = float(_resolve_col(row, sales_map, "revenue", 0) or 0),
                        year       = int(_resolve_col(row, sales_map, "year", now.year) or now.year),
                        month      = int(_resolve_col(row, sales_map, "month", now.month) or now.month),
                        region     = str(_resolve_col(row, sales_map, "region", "") or ""),
                        brand      = str(_resolve_col(row, sales_map, "brand", "") or ""),
                    )
                    db.session.add(sale)
                    total_rows += 1
                except Exception as row_err:
                    logger.warning("Sale row skip: %s", row_err)
                    continue

            db.session.commit()
        except Exception as sale_err:
            db.session.rollback()
            logger.error("Sales table sync failed: %s", sale_err)

        engine.dispose()

        # ── Update sync state ──────────────────────────────────────────────────
        conn_row.sync_status    = "done"
        conn_row.last_synced_at = datetime.utcnow()
        conn_row.last_sync_rows = total_rows
        conn_row.last_error     = None
        db.session.commit()

        logger.info("DB sync done for dealer %d — %d rows imported", dealer_id, total_rows)
        return {"success": True, "rows_imported": total_rows}

    except Exception as e:
        db.session.rollback()
        try:
            conn_row.sync_status = "error"
            conn_row.last_error  = str(e)
            db.session.commit()
        except Exception:
            db.session.rollback()
        logger.error("DB sync FAILED for dealer %d: %s", dealer_id, e)
        return {"success": False, "message": str(e)}


# ── Background auto-sync loop ───────────────────────────────────────────────────

_sync_threads: Dict[int, threading.Thread] = {}
_stop_flags:   Dict[int, threading.Event]  = {}


def start_auto_sync(dealer_id: int, app):
    """
    Spawn a persistent background thread that syncs the dealer's external
    database every `sync_interval_min` minutes.  Safe to call multiple times
    (stops the old thread first).
    """
    stop_auto_sync(dealer_id)   # kill any existing thread first

    stop_event = threading.Event()
    _stop_flags[dealer_id] = stop_event

    def _loop():
        logger.info("Auto-sync thread started for dealer %d", dealer_id)
        while not stop_event.is_set():
            with app.app_context():
                try:
                    from database.models import DealerDatabaseConnection
                    conn_row = DealerDatabaseConnection.query.filter_by(
                        dealer_id=dealer_id, is_active=True
                    ).first()
                    if not conn_row:
                        logger.info("No active connection for dealer %d — stopping sync thread", dealer_id)
                        break

                    interval = (conn_row.sync_interval_min or 30) * 60   # seconds
                    result = sync_dealer_database(dealer_id)
                    logger.info("Auto-sync dealer %d: %s", dealer_id, result)

                    if result.get("success"):
                        # Trigger ML pipeline after a successful sync (wrapped in context)
                        def _run_pipeline_with_context(d_id, a):
                            with a.app_context():
                                from services.analysis_engine import run_full_pipeline_for_dealer
                                run_full_pipeline_for_dealer(d_id)

                        import threading as _t
                        _t.Thread(
                            target=_run_pipeline_with_context,
                            args=(dealer_id, app),
                            daemon=True
                        ).start()
                except Exception as loop_err:
                    logger.error("Auto-sync loop error dealer %d: %s", dealer_id, loop_err)
                    interval = 300  # retry in 5 min on error

            # Sleep in 5-second chunks so we can respond to stop signals quickly
            for _ in range(int(interval / 5)):
                if stop_event.is_set():
                    break
                time.sleep(5)

        logger.info("Auto-sync thread stopped for dealer %d", dealer_id)

    t = threading.Thread(target=_loop, daemon=True, name=f"db-sync-{dealer_id}")
    t.start()
    _sync_threads[dealer_id] = t


def stop_auto_sync(dealer_id: int):
    """Signal the sync thread for this dealer to stop."""
    if dealer_id in _stop_flags:
        _stop_flags[dealer_id].set()
    _sync_threads.pop(dealer_id, None)
    _stop_flags.pop(dealer_id, None)


def init_all_dealer_syncs(app):
    """Called at server startup to resume auto-sync for all active dealers."""
    from database.models import DealerDatabaseConnection
    with app.app_context():
        try:
            active_conns = DealerDatabaseConnection.query.filter_by(is_active=True).all()
            logger.info("Initializing %d active dealer database syncs...", len(active_conns))
            for conn in active_conns:
                start_auto_sync(conn.dealer_id, app)
        except Exception as e:
            logger.error("Failed to initialize dealer syncs: %s", e)
