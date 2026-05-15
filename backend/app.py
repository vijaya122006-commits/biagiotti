"""
=============================================================================
app.py — Cosmetic Intelligence System | Flask Backend (Rebuilt)
=============================================================================
All new endpoints are database-driven via SQLAlchemy.
ML models (.pkl files) are preserved and used.
=============================================================================
"""
from __future__ import annotations

import logging
import os
import sys
import time
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from flask import Flask, Response, jsonify, request
from flask_cors import CORS
from flask_mail import Mail

# ─── Path setup ───────────────────────────────────────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parent
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# ─── Config ───────────────────────────────────────────────────────────────────
from config import config  # noqa: E402

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("app")

# ─── Flask App ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config.from_object(config)
mail = Mail(app)

_CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*")
CORS(
    app,
    resources={r"/*": {
        "origins": _CORS_ORIGINS,
        "methods": ["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Accept", "Origin", "X-Requested-With"],
        "expose_headers": ["Content-Type", "Authorization"],
    }},
    supports_credentials=False,
)

# ─── Database Init ────────────────────────────────────────────────────────────
from database.db import init_db  # noqa: E402
init_db(app)

# ─── ML Service ───────────────────────────────────────────────────────────────
from services.ml_service import (  # noqa: E402
    predict_skin,
    analyze_sentiment,
    detect_harmful,
    get_similar_products,
    forecast_sales,
    svc,
)
logger.info("ML service loaded — %d models ready", len(svc.ready))

# ─── Blueprints — New routes ──────────────────────────────────────────────────
try:
    from routes.auth_routes import auth_bp
    from routes.dashboard_routes import dashboard_bp
    from routes.product_routes import product_routes_bp
    from routes.review_routes import review_routes_bp
    from routes.database_routes import db_connect_bp

    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(dashboard_bp, url_prefix='/api/dashboard')
    app.register_blueprint(product_routes_bp, url_prefix='/api/products')
    app.register_blueprint(review_routes_bp, url_prefix='/api')
    app.register_blueprint(db_connect_bp, url_prefix='/api/db-connect')
    logger.info("New DB-driven blueprints registered.")
except ImportError as e:
    logger.warning("New blueprint import failed: %s", e)

# ─── Blueprints — Legacy routes (preserved) ───────────────────────────────────
try:
    from routes.upload_routes import upload_bp
    from routes.similarity_routes import similarity_bp
    from routes.safety_routes import safety_bp
    from routes.sentiment_routes import sentiment_bp
    from routes.forecast_routes import forecast_bp
    from routes.report_routes import report_bp
    from routes.pipeline_routes import pipeline_bp
    from routes.product_intelligence_routes import product_intelligence_bp

    app.register_blueprint(upload_bp, url_prefix="/api/upload")
    app.register_blueprint(similarity_bp, url_prefix="/api/similarity")
    app.register_blueprint(safety_bp, url_prefix="/api/safety")
    app.register_blueprint(sentiment_bp, url_prefix="/api/skin-analysis")
    app.register_blueprint(forecast_bp, url_prefix="/api/forecast")
    app.register_blueprint(report_bp, url_prefix="/api")
    app.register_blueprint(pipeline_bp, url_prefix="/api/pipeline")
    app.register_blueprint(product_intelligence_bp, url_prefix="/api/products-legacy")
    logger.info("Legacy blueprints registered.")
except ImportError as _e:
    logger.warning("Legacy blueprint import failed (non-fatal): %s", _e)

# ─── Scheduler ────────────────────────────────────────────────────────────────
try:
    from scheduler.daily_pipeline import init_scheduler
    _scheduler = init_scheduler(app)
except Exception as _se:
    logger.warning("Scheduler init failed (non-fatal): %s", _se)

# ─── Live DB Connection Startup ───────────────────────────────────────────────
try:
    from services.database_sync_service import init_all_dealer_syncs
    init_all_dealer_syncs(app)
except Exception as _dbe:
    logger.warning("Live DB sync initialization failed: %s", _dbe)

# ─── Pipeline refresh endpoint ────────────────────────────────────────────────
@app.route('/api/pipeline/refresh', methods=['POST'])
def refresh_pipeline():
    """Manually trigger a pipeline refresh for the current dealer."""
    from utils.auth import decode_token
    from database.models import DashboardCache
    from services.analysis_engine import run_full_pipeline_for_dealer
    import threading

    token = request.headers.get('Authorization', '').replace('Bearer ', '').strip()
    if not token:
        return jsonify({'error': 'No token'}), 401
    try:
        data = decode_token(token)
        dealer_id = data['dealer_id']
    except Exception:
        return jsonify({'error': 'Invalid token'}), 401

    def _run():
        with app.app_context():
            run_full_pipeline_for_dealer(dealer_id)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return jsonify({'success': True, 'message': 'Pipeline refresh started.'})


@app.route('/api/debug/review-check', methods=['GET'])
def debug_review_check():
    """Diagnostic: check review ↔ product ID alignment."""
    from database.models import Review, Product, AnalysisResult
    total_reviews = Review.query.count()
    total_products = Product.query.count()
    sample_products = [p.product_id for p in Product.query.limit(5).all()]
    sample_review_pids = [r.product_id for r in Review.query.limit(10).all()]
    matched = Review.query.filter(Review.product_id.in_(sample_products)).count()
    results_with_recs = AnalysisResult.query.filter(
        AnalysisResult.recommendations_json.isnot(None),
        AnalysisResult.recommendations_json != '[]',
        AnalysisResult.recommendations_json != 'null',
    ).count()
    return jsonify({
        'total_reviews': total_reviews,
        'total_products': total_products,
        'sample_product_ids': sample_products,
        'sample_review_product_ids': sample_review_pids,
        'reviews_matching_sample_products': matched,
        'analysis_results_with_recommendations': results_with_recs,
    })

# ─── Response helpers ─────────────────────────────────────────────────────────

def _ok(data: Any, message: str = "success", status: int = 200) -> Tuple[Response, int]:
    return jsonify({
        "status": "success",
        "message": message,
        "data": data,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }), status


def _err(message: str, status: int = 400, detail: str = "") -> Tuple[Response, int]:
    body: Dict = {
        "status": "error",
        "message": message,
        "timestamp": datetime.utcnow().isoformat() + "Z",
    }
    if detail:
        body["detail"] = detail
    return jsonify(body), status


def require_json(f: Callable) -> Callable:
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not request.is_json:
            return _err("Request must be JSON.", 400)
        if request.get_json(silent=True) is None:
            return _err("Malformed JSON body.", 400)
        return f(*args, **kwargs)
    return wrapper


def timed(f: Callable) -> Callable:
    @wraps(f)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = f(*args, **kwargs)
        ms = round((time.perf_counter() - t0) * 1000, 1)
        if isinstance(result, tuple) and len(result) == 2:
            resp, code = result
            try:
                data = resp.get_json()
                if isinstance(data, dict):
                    data["processing_ms"] = ms
                    resp.data = __import__("json").dumps(data)
            except Exception:
                pass
        return result
    return wrapper

# ─── Error handlers ───────────────────────────────────────────────────────────

@app.errorhandler(400)
def bad_request(exc): return _err("Bad request.", 400, str(exc))

@app.errorhandler(404)
def not_found(exc): return _err(f"Endpoint '{request.path}' not found.", 404)

@app.errorhandler(405)
def method_not_allowed(exc): return _err(f"Method '{request.method}' not allowed.", 405)

@app.errorhandler(500)
def internal_error(exc):
    logger.exception("Internal server error on %s %s", request.method, request.path)
    return _err("Internal server error.", 500, str(exc))

@app.errorhandler(Exception)
def unhandled_exception(exc):
    logger.exception("Unhandled exception: %s", exc)
    return _err("Unexpected server error.", 500, str(exc))

# ─── Request/response logging ─────────────────────────────────────────────────

@app.before_request
def _log_request():
    request._start_time = time.perf_counter()
    logger.info("→  %s  %s", request.method, request.path)

@app.after_request
def _log_response(response):
    ms = round((time.perf_counter() - getattr(request, "_start_time", 0)) * 1000, 1)
    logger.info("←  %s  %s  %d  (%s ms)", request.method, request.path, response.status_code, ms)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    return response

# ─── Core ML endpoints (preserved from original) ──────────────────────────────

@app.route("/predict-skin", methods=["POST"])
@require_json
@timed
def endpoint_predict_skin():
    body = request.get_json()
    text = body.get("text", "")
    if not text or not str(text).strip():
        return _err("'text' field is required and must not be empty.", 400)
    try:
        result = predict_skin(str(text))
        return _ok(result)
    except Exception as exc:
        return _err("Skin-type prediction failed.", 500, str(exc))


@app.route("/sentiment", methods=["POST"])
@require_json
@timed
def endpoint_sentiment():
    body = request.get_json()
    text = body.get("text", "")
    if not text or not str(text).strip():
        return _err("'text' field is required.", 400)
    try:
        result = analyze_sentiment(str(text))
        return _ok(result)
    except Exception as exc:
        return _err("Sentiment analysis failed.", 500, str(exc))


@app.route("/harmful", methods=["POST"])
@require_json
@timed
def endpoint_harmful():
    body = request.get_json()
    product_id = body.get("product_id")
    ingredient_text = body.get("ingredient_text") or body.get("ingredients", "")
    product_name = body.get("product_name", "")

    if not product_id and not str(ingredient_text).strip() and not product_name:
        return _err("Provide at least one of: 'product_id', 'ingredient_text', or 'product_name'.", 400)
    try:
        result = detect_harmful(
            product_id=str(product_id) if product_id is not None else None,
            ingredient_text=str(ingredient_text) if ingredient_text else "",
            product_name=str(product_name) if product_name else "",
        )
        return _ok(result)
    except Exception as exc:
        return _err("Harmful ingredient detection failed.", 500, str(exc))


@app.route("/similar-products", methods=["POST"])
@require_json
@timed
def endpoint_similar_products():
    body = request.get_json()
    product_index = body.get("product_index")
    product_id = body.get("product_id")
    product_name = body.get("product_name")
    top_n = max(1, min(int(body.get("top_n", 5)), 50))

    if product_index is None and not product_id and not product_name:
        return _err("Provide one of: 'product_index', 'product_id', or 'product_name'.", 400)

    if product_index is not None:
        try:
            product_index = int(product_index)
        except (TypeError, ValueError):
            return _err(f"'product_index' must be an integer.", 400)
    try:
        result = get_similar_products(
            product_index=product_index,
            top_n=top_n,
            product_id=str(product_id) if product_id else None,
            product_name=str(product_name) if product_name else None,
        )
        if "error" in result and not result.get("results"):
            return _err(result["error"], 404)
        return _ok(result)
    except Exception as exc:
        return _err("Similar product lookup failed.", 500, str(exc))


@app.route("/forecast", methods=["POST"])
@require_json
@timed
def endpoint_forecast():
    body = request.get_json()
    features = body.get("features")
    steps = body.get("steps", 7)
    steps = max(1, min(int(steps) if steps is not None else 7, 52))
    product_id = body.get("product_id") or (
        features.get("product_id") if isinstance(features, dict) else None
    )
    forecast_horizon = body.get("forecast_horizon")
    if forecast_horizon is not None:
        try:
            forecast_horizon = int(forecast_horizon)
        except (TypeError, ValueError):
            forecast_horizon = None

    # Allow product_id-only requests — ml_service generates synthetic history
    if features is None and product_id:
        try:
            # Look up real category & name from DB so _resolve_image_url picks the correct image
            from database.models import Product as _Product
            _p = _Product.query.filter_by(product_id=str(product_id)).first()
            _raw_category = _p.category if _p else ''
            _product_name = _p.product_name if _p else ''
            result = svc.forecast_sales(
                features=None,
                steps=steps,
                product_id=str(product_id),
                forecast_horizon=forecast_horizon,
                category=_raw_category,
                product_name=_product_name,
            )
            return _ok(result)
        except Exception as exc:
            return _err("Demand forecast failed.", 500, str(exc))

    if features is None:
        return _err("Provide 'features' list/dict or a 'product_id'.", 400)

    if isinstance(features, list):
        try:
            features = [float(v) for v in features]
        except (TypeError, ValueError) as exc:
            return _err("'features' list must contain numeric values only.", 400, str(exc))
    elif isinstance(features, dict):
        sales = features.get("recent_sales", [])
        if not sales:
            return _err("When 'features' is a dict, 'recent_sales' must be a non-empty list.", 400)
        try:
            features["recent_sales"] = [float(v) for v in sales]
        except (TypeError, ValueError) as exc:
            return _err("'recent_sales' must be a list of numeric values.", 400, str(exc))
    else:
        return _err("'features' must be a list or a dict with 'recent_sales'.", 400)

    ml_input = {}
    if isinstance(features, list):
        ml_input["recent_sales"] = features
    else:
        ml_input = dict(features)

    ml_input["product_id"] = product_id
    ml_input.setdefault("product_name", "Unknown")
    ml_input.setdefault("price", 25.0)
    ml_input.setdefault("units_sold", 100.0)

    try:
        result = forecast_sales(features=ml_input)
        return _ok(result)
    except Exception as exc:
        return _err("Demand forecast failed.", 500, str(exc))

# ─── Health + Info ────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def root():
    return jsonify({
        "status": "success",
        "message": "Cosmetic Intelligence API is running.",
        "version": "3.0.0",
        "docs": "/api/ml/info",
        "new_api": {
            "auth": "/api/auth/login | /api/auth/register",
            "dashboard": "/api/dashboard/summary",
            "products": "/api/products",
            "pipeline": "/api/pipeline/refresh",
        }
    })

@app.route("/api/health", methods=["GET"])
def api_health():
    health = svc.health_check()
    http_status = 200 if health["status"] == "ok" else 503

    # Add DB health
    try:
        from database.models import Dealer
        dealer_count = Dealer.query.count()
        health["db_status"] = "ok"
        health["dealer_count"] = dealer_count
    except Exception as e:
        health["db_status"] = f"error: {e}"

    return jsonify(health), http_status

@app.route("/api/ml/info", methods=["GET"])
def api_ml_info():
    info = svc.service_info()
    info["endpoints"] = [
        {"method": "POST", "path": "/predict-skin", "description": "Skin-type prediction"},
        {"method": "POST", "path": "/sentiment", "description": "Sentiment analysis"},
        {"method": "POST", "path": "/harmful", "description": "Harmful ingredient detection"},
        {"method": "POST", "path": "/similar-products", "description": "Similar product lookup"},
        {"method": "POST", "path": "/forecast", "description": "Demand forecast"},
        {"method": "GET", "path": "/api/health", "description": "Health check"},
        {"method": "GET", "path": "/api/dashboard/summary", "description": "Dashboard KPIs"},
        {"method": "GET", "path": "/api/products", "description": "Product list"},
    ]
    return _ok(info)

# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    os.makedirs(str(_BACKEND_DIR / "data"), exist_ok=True)
    os.makedirs(str(_BACKEND_DIR / "models"), exist_ok=True)
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=False)
