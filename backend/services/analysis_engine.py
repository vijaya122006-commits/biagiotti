"""
services/analysis_engine.py — Batch ML pipeline for all products of a dealer.
Reads from DB, runs ML models, writes results back to DB.
"""
from __future__ import annotations
import json
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any

import numpy as np

_BACKEND = Path(__file__).resolve().parent.parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from database.models import db, Product, Inventory, Sale, Review, AnalysisResult, DashboardCache, Notification
from services.notification_service import create_notification, clear_old_notifications

# Import ML functions
from services.ml_service import predict_skin, analyze_sentiment, detect_harmful, get_similar_products, forecast_sales

logger = logging.getLogger("analysis_engine")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(h)
    logger.setLevel(logging.INFO)

# In-memory dict still used within a single pipeline run for fast incremental updates.
# Bug 7 fix: get_pipeline_progress() reads from DashboardCache so multi-worker
# deployments (gunicorn, containers) all see the same state.
_pipeline_progress: Dict[int, Dict] = {}


def get_pipeline_progress(dealer_id: int) -> Dict:
    """Return current pipeline progress, preferring the DB-backed cache for multi-worker safety."""
    try:
        from database.models import DashboardCache
        cache = DashboardCache.query.filter_by(dealer_id=dealer_id).first()
        if cache:
            return {
                'status': cache.pipeline_status or 'idle',
                'progress_pct': cache.pipeline_progress or 0,
                'products_processed': 0,   # not stored in cache; use progress_pct as proxy
                'total_products': cache.total_products or 0,
                'started_at': (
                    cache.pipeline_started_at.isoformat() + 'Z'
                    if getattr(cache, 'pipeline_started_at', None) else None
                ),
                'last_updated': (
                    cache.last_updated.isoformat() + 'Z'
                    if cache.last_updated else None
                ),
            }
    except Exception:
        pass  # Fall back to in-memory dict if DB is unavailable
    return _pipeline_progress.get(dealer_id, {
        'status': 'idle', 'progress_pct': 0,
        'products_processed': 0, 'total_products': 0,
        'started_at': None,
    })


def _get_monthly_sales(product_id: str, dealer_id: int) -> list:
    """
    Fetch sales data for a product. Looks back up to 36 months to find real data.
    Returns a 12-month list of monthly unit totals (most recent last).
    """
    from sqlalchemy import func
    now = datetime.utcnow()

    # Try to get ALL sales records for this product, regardless of date
    all_sales = Sale.query.filter_by(
        product_id=product_id, dealer_id=dealer_id
    ).filter(
        Sale.year.isnot(None), Sale.month.isnot(None), Sale.units_sold > 0
    ).order_by(Sale.year, Sale.month).all()

    if not all_sales:
        # No sales data — return zeros to reflect the actual database state
        return [0.0] * 12

    # Build a lookup of {(year, month): units}
    sales_map = {}
    for s in all_sales:
        key = (s.year, s.month)
        sales_map[key] = sales_map.get(key, 0) + (s.units_sold or 0)

    # Build last 12 months from now
    monthly = []
    for i in range(11, -1, -1):
        dt = now - timedelta(days=30 * i)
        yr, mo = dt.year, dt.month
        monthly.append(sales_map.get((yr, mo), 0.0))

    # If last 12 months has NO data, scale from historical data
    non_zero_recent = [v for v in monthly if v > 0]
    if not non_zero_recent:
        # Use historical average as a proxy for current velocity
        hist_values = list(sales_map.values())
        hist_avg = float(np.mean(hist_values)) if hist_values else 10.0
        # Scale down — historical data may be annual totals or from a different period
        # Use it directly as monthly estimate
        monthly = [hist_avg] * 12
    else:
        # Fill any zero months with rolling average of non-zero months
        avg = float(np.mean(non_zero_recent))
        monthly = [v if v > 0 else avg for v in monthly]

    return monthly


def _compute_stock_status(current_stock: float, monthly_sales: list) -> Dict:
    """Calculate stock status from current stock and recent sales velocity."""
    last3 = monthly_sales[-3:] if len(monthly_sales) >= 3 else monthly_sales
    avg_monthly = float(np.mean(last3)) if last3 else 10.0
    # Guard against unrealistically high averages from bulk historical data
    # Cap at a sensible max (e.g. 5000 units/month for cosmetics)
    avg_monthly = min(avg_monthly, 5000.0)
    avg_monthly = max(avg_monthly, 1.0)  # at least 1 unit/month
    daily_rate = avg_monthly / 30.0

    days_until_stockout = current_stock / daily_rate

    # Thresholds tuned for cosmetics retail:
    # <14 days = critical understock
    # 14-30 = low stock warning
    # 30-180 = normal (up to 6 months supply is fine)
    # >180 days = overstock
    if days_until_stockout < 7:
        stock_status = 'understock'
        stockout_risk = 'HIGH'
    elif days_until_stockout < 30:
        stock_status = 'understock'
        stockout_risk = 'MEDIUM'
    elif days_until_stockout > 180:
        stock_status = 'overstock'
        stockout_risk = 'LOW'
    elif days_until_stockout > 90:
        stock_status = 'watch'
        stockout_risk = 'LOW'
    else:
        stock_status = 'normal'
        stockout_risk = 'LOW'

    return {
        'stock_status': stock_status,
        'stockout_risk': stockout_risk,
        'days_until_stockout': round(days_until_stockout, 1),
        'daily_rate': round(daily_rate, 2),
        'avg_monthly': round(avg_monthly, 1),
    }


def _compute_stock_decision(days_until_stockout: float, stock_status: str,
                             risk_level: str, avg_monthly: float,
                             forecast_trend: str) -> tuple:
    """
    Returns (decision: str, reason: str, priority_score: float).
    Rule-based logic that correctly reflects urgency.
    """
    # Rule 1: HIGH harmful — override everything
    if risk_level == 'high':
        return (
            'REMOVE FROM SALE',
            'Product contains HIGH risk harmful ingredients. Immediate removal recommended '
            'to protect customers and avoid liability.',
            100.0
        )

    # Rule 2: No sales data
    if avg_monthly <= 0:
        if stock_status == 'understock':
            return ('INVESTIGATE', 'Stock is low but no sales history found. Confirm if product is new or discontinued.', 40.0)
        return ('MONITOR', 'No sales history available. Add sales data to get accurate decisions.', 20.0)

    days = days_until_stockout if days_until_stockout is not None else 9999
    reorder_3m = max(int(avg_monthly * 3), 50)
    reorder_2m = max(int(avg_monthly * 2), 30)

    # Rule 3: Critical < 3 days
    if days < 3:
        return (
            'AGGRESSIVE RESTOCK',
            f'CRITICAL: Only {round(days, 1)} day(s) of stock remaining at current sales rate. '
            f'Order at least {reorder_3m} units immediately to cover 3 months of demand.',
            95.0
        )

    # Rule 4: Urgent 3–7 days
    if days < 7:
        return (
            'RESTOCK URGENTLY',
            f'Stock will run out in {round(days, 1)} days. '
            f'Order {reorder_2m} units within 48 hours to avoid stockout.',
            85.0
        )

    # Rule 5: Low 7–14 days
    if days < 14:
        return (
            'RESTOCK SOON',
            f'{round(days)} days of stock left. '
            f'Place order for {reorder_2m} units within the week.',
            70.0
        )

    # Rule 6: Medium harmful
    if risk_level == 'medium':
        return (
            'CAUTION — DO NOT RESTOCK',
            'Product contains medium-risk ingredients. Do not order more stock. '
            'Evaluate safety report before next purchase order.',
            65.0
        )

    # Rule 7: Overstock
    if stock_status == 'overstock':
        if forecast_trend == 'decreasing':
            return (
                'REDUCE PRICE — CLEAR STOCK',
                f'{round(days)} days of supply on hand and demand is declining. '
                f'Offer 15-25% discount to clear inventory.',
                60.0
            )
        return (
            'HOLD — MONITOR SALES',
            f'{round(days)} days of supply on hand. Sales are stable. '
            f'Do not reorder until stock drops below {int(avg_monthly * 1.5)} units.',
            35.0
        )

    # Rule 8: Trending + normal
    if forecast_trend == 'increasing' and stock_status == 'normal':
        return (
            'INCREASE STOCK',
            f'Demand is trending upward. Consider ordering {int(avg_monthly * 1.5)} extra units to prepare for surge.',
            50.0
        )

    # Rule 9: Normal / healthy
    return (
        'MAINTAIN STOCK',
        f'Stock levels healthy with {round(days)} days of supply. '
        f'Reorder when stock drops below {int(avg_monthly * 1.5)} units.',
        25.0
    )


def _compute_verification(product: Product, safety_score: float, risk_level: str,
                           review_count: int, avg_rating: float, total_sales: float,
                           skin_type: str) -> str:
    checks = {
        'has_ingredients': bool(product.ingredients and len(product.ingredients) > 10),
        'passed_safety': safety_score >= 60,
        'has_skin_type': bool(skin_type),
        'has_reviews': review_count >= 3,
        'good_rating': avg_rating >= 3.5,
        'has_sales': total_sales > 0,
        'forecast_done': True,
        'not_harmful_high': risk_level != 'high',
    }
    passed = sum(checks.values())

    if passed == 8:
        return 'fully_verified'
    elif passed >= 6:
        return 'partial'
    elif risk_level == 'high':
        return 'flagged'
    else:
        return 'unverified'


def _upsert_analysis(result_data: Dict, product_id: str, dealer_id: int):
    """Upsert into analysis_results table."""
    try:
        existing = AnalysisResult.query.filter_by(
            product_id=product_id, dealer_id=dealer_id
        ).first()

        if existing:
            for k, v in result_data.items():
                if hasattr(existing, k):
                    setattr(existing, k, v)
            existing.analyzed_at = datetime.utcnow()
        else:
            row = AnalysisResult(
                product_id=product_id,
                dealer_id=dealer_id,
                **{k: v for k, v in result_data.items() if hasattr(AnalysisResult, k)},
            )
            db.session.add(row)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error("Upsert failed for %s: %s", product_id, e)


def run_full_pipeline_for_dealer(dealer_id: int) -> Dict[str, Any]:
    """
    Run complete ML analysis for all products of a given dealer.
    Updates dashboard cache and notifications on completion.
    """
    logger.info("=" * 60)
    logger.info("Starting pipeline for dealer %d", dealer_id)
    logger.info("=" * 60)

    # Mark pipeline as running
    _pipeline_progress[dealer_id] = {
        'status': 'running',
        'progress_pct': 0,
        'products_processed': 0,
        'total_products': 0,
        'started_at': datetime.utcnow().isoformat() + 'Z',
    }

    # Update DashboardCache status
    try:
        cache = DashboardCache.query.filter_by(dealer_id=dealer_id).first()
        if not cache:
            cache = DashboardCache(dealer_id=dealer_id)
            db.session.add(cache)
        cache.pipeline_status = 'running'
        cache.pipeline_started_at = datetime.utcnow()
        cache.pipeline_progress = 0
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error("Failed to update cache status: %s", e)

    products = Product.query.filter_by(dealer_id=dealer_id).all()
    total = len(products)
    _pipeline_progress[dealer_id]['total_products'] = total

    if total == 0:
        logger.warning("No products found for dealer %d", dealer_id)
        _pipeline_progress[dealer_id]['status'] = 'done'
        return {'products_analyzed': 0}

    logger.info("Processing %d products for dealer %d", total, dealer_id)

    stats = {
        'understock': 0, 'overstock': 0,
        'harmful_high': 0, 'harmful_medium': 0,
        'total_sales': 0.0,
        'top_products': [],
        'category_stats': {},
    }

    # Clear old notifications before fresh run
    clear_old_notifications(dealer_id, days=7)

    for idx, product in enumerate(products):
        try:
            pid = product.product_id

            # ── Step A: Load sales and inventory ──────────────────────────────
            monthly_sales = _get_monthly_sales(pid, dealer_id)
            inv = Inventory.query.filter_by(product_id=pid, dealer_id=dealer_id).first()
            current_stock = inv.current_stock if inv else 50.0
            lead_time = inv.lead_time_days if inv else 14.0

            # ── Step B: Demand Forecast ────────────────────────────────────────
            forecast_input = {
                'recent_sales': monthly_sales,
                'product_id': pid,
                'product_name': product.product_name,
                'price': product.price or 500.0,
                'units_sold': np.mean(monthly_sales),
                'current_stock': current_stock,
                'lead_time_days': lead_time,
                'cost_price': product.cost_price or (product.price or 500.0) * 0.4,
            }
            fc_result = forecast_sales(features=forecast_input)
            forecast_json = json.dumps(fc_result.get('forecast', []))
            forecast_trend = fc_result.get('trend', 'stable')
            stock_decision_raw = fc_result.get('decision', 'MAINTAIN STOCK')
            # strip priority score suffix
            stock_decision = stock_decision_raw.split('|')[0].strip()
            decision_reason = fc_result.get('reason', '')

            # ── Step C: Skin Type Detection ────────────────────────────────────
            skin_text = product.ingredients or product.product_name
            skin_result = predict_skin(skin_text)
            skin_type = skin_result.get('skin_type', '')
            skin_confidence = skin_result.get('confidence', 0.0)

            # Update product skin_suitability if empty
            if not product.skin_suitability and skin_type:
                product.skin_suitability = skin_type
                try:
                    db.session.commit()
                except Exception:
                    db.session.rollback()

            # ── Step D: Harmful Ingredient Detection ───────────────────────────
            harmful_result = detect_harmful(
                product_id=pid,
                ingredient_text=product.ingredients or '',
                product_name=product.product_name,
            )
            safety_score = harmful_result.get('safety_score', 100.0)
            safety_status = harmful_result.get('status', 'Safe')
            harmful_json = json.dumps(harmful_result.get('harmful_ingredients', []))
            risk_level = 'high' if safety_score < 60 else ('medium' if safety_score < 80 else 'none')

            # ── Step E: Stock Status ────────────────────────────────────────────
            stock_info = _compute_stock_status(current_stock, monthly_sales)
            if stock_info['stock_status'] in ('understock',):
                stats['understock'] += 1
            elif stock_info['stock_status'] == 'overstock':
                stats['overstock'] += 1
            if risk_level in ('high', 'medium'):
                stats['harmful_high' if risk_level == 'high' else 'harmful_medium'] += 1

            # ── Step E2: Smart Stock Decision (overrides ML default) ───────────
            stock_decision, decision_reason, priority_score = _compute_stock_decision(
                days_until_stockout=stock_info['days_until_stockout'],
                stock_status=stock_info['stock_status'],
                risk_level=risk_level,
                avg_monthly=stock_info['avg_monthly'],
                forecast_trend=forecast_trend,
            )

            # ── Step F: Review Aggregation ─────────────────────────────────────
            reviews = Review.query.filter_by(product_id=pid, dealer_id=dealer_id).all()
            review_count = len(reviews)
            avg_rating = float(np.mean([r.rating for r in reviews if r.rating])) if reviews else 0.0
            sentiment_avg = float(np.mean([r.sentiment_score for r in reviews if r.sentiment_score is not None])) if reviews else 0.0

            # ── Step G: Verification Status ───────────────────────────────────
            total_sales_product = sum(monthly_sales)
            stats['total_sales'] += total_sales_product
            verification = _compute_verification(
                product, safety_score, risk_level, review_count, avg_rating, total_sales_product, skin_type
            )

            # ── Step H: Similar Products ──────────────────────────────────────
            sim_result = get_similar_products(product_id=pid, product_name=product.product_name, top_n=5)
            sim_json = json.dumps([r.get('product_id') for r in sim_result.get('results', [])])

            # ── Step I: Write to DB ───────────────────────────────────────────
            result_data = {
                'demand_forecast_json': forecast_json,
                'forecast_trend': forecast_trend,
                'stock_status': stock_info['stock_status'],
                'days_until_stockout': stock_info['days_until_stockout'],
                'stockout_risk': stock_info['stockout_risk'],
                'skin_type_detected': skin_type,
                'skin_confidence': round(skin_confidence, 3),
                'harmful_ingredients_json': harmful_json,
                'safety_score': round(safety_score, 1),
                'safety_status': safety_status,
                'risk_level': risk_level,
                'stock_decision': stock_decision[:100],
                'decision_reason': decision_reason,
                'priority_score': priority_score,
                'recommendations_json': sim_json,
                'avg_rating': round(avg_rating, 2),
                'review_count': review_count,
                'sentiment_avg': round(sentiment_avg, 3),
                'verification_status': verification,
            }
            _upsert_analysis(result_data, pid, dealer_id)

            # Track for dashboard
            stats['top_products'].append({
                'product_id': pid,
                'product_name': product.product_name,
                'category': product.category,
                'brand': product.brand,
                'price': product.price,
                'total_sales': round(total_sales_product, 1),
                'avg_rating': round(avg_rating, 2),
                'stock_status': stock_info['stock_status'],
                'stockout_risk': stock_info['stockout_risk'],
                'days_until_stockout': stock_info['days_until_stockout'],
                'risk_level': risk_level,
                'forecast_trend': forecast_trend,
                'priority_score': priority_score,
            })

            cat = product.category or 'Other'
            if cat not in stats['category_stats']:
                stats['category_stats'][cat] = {'total': 0, 'healthy': 0}
            stats['category_stats'][cat]['total'] += 1
            if stock_info['stock_status'] == 'normal' and risk_level == 'none':
                stats['category_stats'][cat]['healthy'] += 1

            # Update progress
            pct = round((idx + 1) / total * 100)
            _pipeline_progress[dealer_id]['products_processed'] = idx + 1
            _pipeline_progress[dealer_id]['progress_pct'] = pct

            # Update cache progress periodically
            if (idx + 1) % 50 == 0:
                try:
                    c = DashboardCache.query.filter_by(dealer_id=dealer_id).first()
                    if c:
                        c.pipeline_progress = pct
                        db.session.commit()
                except Exception:
                    db.session.rollback()

        except Exception as e:
            logger.error("Pipeline error on product %s: %s", product.product_id, e)
            continue

    # ── Step J: Generate Notifications ────────────────────────────────────────
    _generate_notifications(dealer_id, stats['top_products'])

    # ── Step K: Update Dashboard Cache ────────────────────────────────────────
    # Prioritize: Increasing Trend > High Priority Score > Total Sales
    sorted_products = sorted(stats['top_products'], key=lambda x: (
        1 if x['forecast_trend'] == 'increasing' else 0,
        x['priority_score'],
        x['total_sales']
    ), reverse=True)
    top10 = sorted_products[:10]
    alerts = _build_alert_feed(dealer_id)
    cat_health = {
        cat: {
            'total': v['total'],
            'healthy': v['healthy'],
            'health_pct': round(v['healthy'] / max(1, v['total']) * 100),
        }
        for cat, v in stats['category_stats'].items()
    }

    critical_alerts = sum(1 for p in stats['top_products']
                         if p['stockout_risk'] == 'HIGH' or p['risk_level'] == 'high')

    try:
        cache = DashboardCache.query.filter_by(dealer_id=dealer_id).first()
        if not cache:
            cache = DashboardCache(dealer_id=dealer_id)
            db.session.add(cache)

        cache.total_products = total
        cache.understock_count = stats['understock']
        cache.overstock_count = stats['overstock']
        cache.harmful_count = stats['harmful_high'] + stats['harmful_medium']
        cache.critical_alerts = critical_alerts
        cache.top_products_json = json.dumps(top10)
        cache.alert_feed_json = json.dumps(alerts)
        cache.category_health_json = json.dumps(cat_health)
        cache.pipeline_status = 'done'
        cache.pipeline_progress = 100
        cache.last_updated = datetime.utcnow()
        cache.next_scheduled = datetime.utcnow().replace(hour=2, minute=0, second=0) + timedelta(days=1)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        logger.error("Failed to update dashboard cache: %s", e)
        try:
            cache = DashboardCache.query.filter_by(dealer_id=dealer_id).first()
            if cache:
                cache.pipeline_status = 'failed'
                db.session.commit()
        except Exception:
            db.session.rollback()

    _pipeline_progress[dealer_id] = {
        'status': 'done',
        'progress_pct': 100,
        'products_processed': total,
        'total_products': total,
        'started_at': _pipeline_progress[dealer_id].get('started_at'),
    }

    logger.info("Pipeline complete for dealer %d — %d products analyzed", dealer_id, total)
    return {
        'products_analyzed': total,
        'understock': stats['understock'],
        'overstock': stats['overstock'],
        'harmful': stats['harmful_high'] + stats['harmful_medium'],
    }


def _generate_notifications(dealer_id: int, products: list):
    """Generate smart notifications from analysis results."""
    # Sort by priority
    critical_stock = [p for p in products if p['stockout_risk'] == 'HIGH']
    medium_stock = [p for p in products if p['stockout_risk'] == 'MEDIUM']
    harmful_high = [p for p in products if p['risk_level'] == 'high']
    harmful_medium = [p for p in products if p['risk_level'] == 'medium']
    overstock = [p for p in products if p['stock_status'] == 'overstock']
    surge = sorted(
        [p for p in products if p['forecast_trend'] == 'increasing'],
        key=lambda x: x['priority_score'], reverse=True
    )[:5]

    for p in critical_stock:
        create_notification(
            dealer_id=dealer_id,
            product_id=p['product_id'],
            notif_type='understock',
            title=f"⚠️ Critical Stockout: {p['product_name'][:40]}",
            message=f"Will stock out in {p['days_until_stockout']:.0f} days. Restock immediately.",
            severity='critical',
        )

    for p in medium_stock:
        create_notification(
            dealer_id=dealer_id,
            product_id=p['product_id'],
            notif_type='understock',
            title=f"Low Stock: {p['product_name'][:40]}",
            message=f"{p['days_until_stockout']:.0f} days of inventory remaining.",
            severity='high',
        )

    for p in harmful_high[:10]:
        create_notification(
            dealer_id=dealer_id,
            product_id=p['product_id'],
            notif_type='harmful',
            title=f"☠️ Harmful Ingredients: {p['product_name'][:40]}",
            message="Product contains HIGH risk ingredients. Consider removing from inventory.",
            severity='critical',
        )

    for p in harmful_medium[:10]:
        create_notification(
            dealer_id=dealer_id,
            product_id=p['product_id'],
            notif_type='harmful',
            title=f"⚠️ Safety Warning: {p['product_name'][:40]}",
            message="Product contains MEDIUM risk ingredients. Review before promoting.",
            severity='medium',
        )

    for p in overstock[:10]:
        create_notification(
            dealer_id=dealer_id,
            product_id=p['product_id'],
            notif_type='overstock',
            title=f"Overstock: {p['product_name'][:40]}",
            message=f"Overstocked by {p['days_until_stockout']:.0f} days. Consider promotions.",
            severity='low',
        )

    for p in surge:
        create_notification(
            dealer_id=dealer_id,
            product_id=p['product_id'],
            notif_type='surge',
            title=f"📈 Demand Surge: {p['product_name'][:40]}",
            message="Demand is increasing. Ensure adequate stock levels.",
            severity='medium',
        )


def _build_alert_feed(dealer_id: int) -> list:
    """Build the prioritized alert feed for the dashboard."""
    from database.models import Notification
    notifs = Notification.query.filter_by(
        dealer_id=dealer_id, is_read=False
    ).order_by(Notification.created_at.desc()).all()

    severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
    notifs.sort(key=lambda n: (severity_order.get(n.severity, 4), -n.notif_id))

    return [{
        'id': n.notif_id,
        'type': n.notif_type,
        'title': n.title,
        'message': n.message,
        'product_id': n.product_id,
        'severity': n.severity,
        'created_at': n.created_at.isoformat() + 'Z' if n.created_at else None,
    } for n in notifs[:50]]
