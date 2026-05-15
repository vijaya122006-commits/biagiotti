"""
routes/dashboard_routes.py — Dashboard summary, alerts (top-10 prioritized), pipeline status
"""
import json
import logging
from datetime import datetime
from flask import Blueprint, jsonify, request
from database.models import db, DashboardCache, Notification, Product, Inventory, AnalysisResult, Sale
from services.notification_service import mark_notification_read, get_dealer_notifications
from services.analysis_engine import get_pipeline_progress
from utils.auth import require_auth

dashboard_bp = Blueprint('dashboard', __name__)
logger = logging.getLogger("dashboard_routes")


# ─── Alert Feed Builder ────────────────────────────────────────────────────────

def build_alert_feed(dealer_id: int, limit: int = 10) -> list:
    """
    Build top-N prioritized alerts, queried fresh each time.
    Priority: 1=harmful-high, 2=critical-stockout(<3d), 3=urgent-stockout(<7d),
              4=medium-harmful, 5=demand-surge, 6=overstock
    """
    alerts = []

    # --- Priority 1: HIGH harmful ingredients ---
    try:
        harmful_high = db.session.query(Product, AnalysisResult).join(
            AnalysisResult,
            (AnalysisResult.product_id == Product.product_id) &
            (AnalysisResult.dealer_id == Product.dealer_id)
        ).filter(
            Product.dealer_id == dealer_id,
            AnalysisResult.risk_level == 'high'
        ).limit(5).all()

        for product, analysis in harmful_high:
            chemicals = []
            if analysis.harmful_ingredients_json:
                try:
                    raw = json.loads(analysis.harmful_ingredients_json)
                    chemicals = [
                        (h.get('name', h) if isinstance(h, dict) else str(h))
                        for h in raw[:2]
                    ]
                except Exception:
                    pass
            chem_str = ', '.join(chemicals) if chemicals else 'harmful ingredients'
            alerts.append({
                'type': 'harmful', 'severity': 'critical', 'priority': 1,
                'product_id': product.product_id,
                'title': f'☠️ Harmful: {product.product_name[:45]}',
                'message': f'Contains {chem_str}. Consider removing from sale.',
                'action': 'Remove from sale', 'icon': '☠️',
            })
    except Exception as e:
        logger.warning("Alert build error (harmful_high): %s", e)

    # --- Priority 2: CRITICAL stockout < 3 days ---
    try:
        critical_stock = db.session.query(Product, AnalysisResult, Inventory).join(
            AnalysisResult,
            (AnalysisResult.product_id == Product.product_id) &
            (AnalysisResult.dealer_id == Product.dealer_id)
        ).join(
            Inventory,
            (Inventory.product_id == Product.product_id) &
            (Inventory.dealer_id == Product.dealer_id)
        ).filter(
            Product.dealer_id == dealer_id,
            AnalysisResult.days_until_stockout < 3,
            AnalysisResult.days_until_stockout >= 0
        ).order_by(AnalysisResult.days_until_stockout.asc()).limit(5).all()

        for product, analysis, inventory in critical_stock:
            days = round(float(analysis.days_until_stockout or 0), 1)
            stock = int(inventory.current_stock or 0)
            alerts.append({
                'type': 'understock', 'severity': 'critical', 'priority': 2,
                'product_id': product.product_id,
                'title': f'🔴 Critical Stock: {product.product_name[:45]}',
                'message': f'Only {stock} units — stocks out in {days}d. Order immediately.',
                'action': 'Restock now', 'icon': '🔴', 'days_left': days,
            })
    except Exception as e:
        logger.warning("Alert build error (critical_stock): %s", e)

    # --- Priority 3: Urgent stockout 3–7 days ---
    try:
        urgent_stock = db.session.query(Product, AnalysisResult, Inventory).join(
            AnalysisResult,
            (AnalysisResult.product_id == Product.product_id) &
            (AnalysisResult.dealer_id == Product.dealer_id)
        ).join(
            Inventory,
            (Inventory.product_id == Product.product_id) &
            (Inventory.dealer_id == Product.dealer_id)
        ).filter(
            Product.dealer_id == dealer_id,
            AnalysisResult.days_until_stockout >= 3,
            AnalysisResult.days_until_stockout < 7
        ).order_by(AnalysisResult.days_until_stockout.asc()).limit(3).all()

        for product, analysis, inventory in urgent_stock:
            days = round(float(analysis.days_until_stockout or 0), 1)
            stock = int(inventory.current_stock or 0)
            alerts.append({
                'type': 'understock', 'severity': 'warning', 'priority': 3,
                'product_id': product.product_id,
                'title': f'🟠 Low Stock: {product.product_name[:45]}',
                'message': f'{stock} units left — {days}d of supply. Plan restock soon.',
                'action': 'Plan restock', 'icon': '🟠', 'days_left': days,
            })
    except Exception as e:
        logger.warning("Alert build error (urgent_stock): %s", e)

    # --- Priority 4: MEDIUM harmful ---
    try:
        harmful_med = db.session.query(Product, AnalysisResult).join(
            AnalysisResult,
            (AnalysisResult.product_id == Product.product_id) &
            (AnalysisResult.dealer_id == Product.dealer_id)
        ).filter(
            Product.dealer_id == dealer_id,
            AnalysisResult.risk_level == 'medium'
        ).limit(3).all()

        for product, analysis in harmful_med:
            alerts.append({
                'type': 'harmful', 'severity': 'medium', 'priority': 4,
                'product_id': product.product_id,
                'title': f'⚠️ Safety Warning: {product.product_name[:45]}',
                'message': 'Medium-risk ingredients detected. Do not reorder until reviewed.',
                'action': 'Review ingredients', 'icon': '⚠️',
            })
    except Exception as e:
        logger.warning("Alert build error (harmful_med): %s", e)

    # --- Priority 5: Demand surge ---
    try:
        trending = db.session.query(Product, AnalysisResult).join(
            AnalysisResult,
            (AnalysisResult.product_id == Product.product_id) &
            (AnalysisResult.dealer_id == Product.dealer_id)
        ).filter(
            Product.dealer_id == dealer_id,
            AnalysisResult.forecast_trend == 'increasing',
            AnalysisResult.stock_status != 'understock'
        ).limit(3).all()

        for product, analysis in trending:
            alerts.append({
                'type': 'opportunity', 'severity': 'info', 'priority': 5,
                'product_id': product.product_id,
                'title': f'📈 Demand Rising: {product.product_name[:45]}',
                'message': 'Sales trend increasing. Stock up to capture demand.',
                'action': 'View forecast', 'icon': '📈',
            })
    except Exception as e:
        logger.warning("Alert build error (trending): %s", e)

    # --- Priority 6: Overstock ---
    try:
        overstock = db.session.query(Product, AnalysisResult).join(
            AnalysisResult,
            (AnalysisResult.product_id == Product.product_id) &
            (AnalysisResult.dealer_id == Product.dealer_id)
        ).filter(
            Product.dealer_id == dealer_id,
            AnalysisResult.stock_status == 'overstock'
        ).limit(3).all()

        for product, analysis in overstock:
            days = round(float(analysis.days_until_stockout or 0))
            alerts.append({
                'type': 'overstock', 'severity': 'warning', 'priority': 6,
                'product_id': product.product_id,
                'title': f'🟡 Excess Stock: {product.product_name[:45]}',
                'message': f'{days}d of supply on hand. Consider discounts to clear stock.',
                'action': 'Consider discount', 'icon': '🟡',
            })
    except Exception as e:
        logger.warning("Alert build error (overstock): %s", e)

    # --- Add Recent Notifications from DB ---
    try:
        from services.notification_service import get_dealer_notifications
        unread = get_dealer_notifications(dealer_id, unread_only=True)
        for n in unread[:5]:
            # Avoid duplicating if already in alerts (by product_id)
            if not any(a.get('product_id') == n['product_id'] and a.get('type') == n['type'] for a in alerts):
                alerts.append({
                    'id': n['id'],
                    'type': n['type'], 'severity': n['severity'], 'priority': 0, # Priority 0 for explicit notifications
                    'product_id': n['product_id'],
                    'title': n['title'],
                    'message': n['message'],
                    'action': 'View detail', 'icon': '🔔' if n['severity'] == 'medium' else '⚠️',
                })
    except Exception as e:
        logger.warning("Alert build error (notifications): %s", e)

    alerts.sort(key=lambda x: x['priority'])
    return alerts[:limit]


# ─── Category Health ──────────────────────────────────────────────────────────

def compute_category_health_live(dealer_id: int) -> dict:
    """Compute fresh category health percentages directly from DB."""
    try:
        from sqlalchemy import case, func
        results = db.session.query(
            Product.category,
            func.count(Product.product_id).label('total'),
            func.sum(
                case(
                    (
                        (AnalysisResult.stock_status == 'normal') &
                        (AnalysisResult.risk_level.in_(['none', 'low'])),
                        1
                    ),
                    else_=0
                )
            ).label('healthy')
        ).outerjoin(
            AnalysisResult,
            (AnalysisResult.product_id == Product.product_id) &
            (AnalysisResult.dealer_id == Product.dealer_id)
        ).filter(
            Product.dealer_id == dealer_id,
            Product.category.isnot(None),
            Product.category != ''
        ).group_by(Product.category).all()

        health = {}
        for row in results:
            cat = (row.category or '').strip() or 'Unknown'
            total = int(row.total or 0)
            healthy = int(row.healthy or 0)
            pct = round(healthy / total * 100, 1) if total > 0 else 0.0
            health[cat] = {'total': total, 'healthy': healthy, 'health_pct': pct}

        # Sort most problematic first
        return dict(sorted(health.items(), key=lambda x: x[1]['health_pct']))
    except Exception as e:
        logger.error("compute_category_health_live error: %s", e)
        return {}


# ─── Routes ──────────────────────────────────────────────────────────────────

@dashboard_bp.route('/summary', methods=['GET'])
@require_auth
def get_dashboard_summary():
    """GET /api/dashboard/summary — returns KPIs, fresh alerts, top products."""
    dealer_id = request.dealer_id
    horizon = request.args.get('forecast_horizon', '30')
    cache = DashboardCache.query.filter_by(dealer_id=dealer_id).first()

    # If cache is missing or very old (> 1 hour), refresh it
    should_refresh = not cache
    if cache and cache.last_updated:
        elapsed = (datetime.utcnow() - cache.last_updated).total_seconds()
        if elapsed > 3600: # 1 hour
            should_refresh = True
            
    if should_refresh:
        # If cache is missing, we at least want to ensure a record exists
        if not cache:
            cache = DashboardCache(dealer_id=dealer_id, pipeline_status='pending', last_updated=datetime.utcnow())
            db.session.add(cache)
            db.session.commit()

    # Import shared image helper (unique per product_id)
    from routes.product_routes import _product_image_url

    # 1. Top Products — horizon-aware scoring (different products per window)
    try:
        horizon_int = int(horizon) if str(horizon).isdigit() else 30
        from sqlalchemy import desc as _desc, asc as _asc

        # --- Horizon-specific focus bands ---
        # Short  (≤7d) : URGENT — products running out right now
        # Medium (≤30d): PLAN   — products running out this month + rising demand
        # Long   (>30d) : OPPORTUNITY — rising trend + overstock clearance + strategic
        if horizon_int <= 1:
            band_min, band_max = 0, 1
        elif horizon_int <= 7:
            band_min, band_max = 0, 7
        elif horizon_int <= 21:
            band_min, band_max = 3, 21      # skip the 0-3d (already handled in 7d)
        elif horizon_int <= 30:
            band_min, band_max = 7, 30      # skip the urgent ones, plan ahead
        elif horizon_int <= 90:
            band_min, band_max = 14, 90     # medium-term stockout + rising trend
        else:
            band_min, band_max = 30, 180    # long-term opportunity

        def _horizon_score(a, horizon_int: int) -> float:
            """Score a product's relevance for the given forecast window."""
            if not a:
                return 0.0
            score = 0.0
            days = float(a.days_until_stockout) if a.days_until_stockout is not None else 999.0
            trend = (a.forecast_trend or 'stable').lower()
            status = (a.stock_status or 'normal').lower()
            risk = (a.stockout_risk or 'LOW').upper()
            rating = float(a.avg_rating or 0)

            # --- Stockout urgency relative to THIS window ---
            if band_min <= days <= band_max:
                # How deep into the window does this stockout fall?
                window_size = max(band_max - band_min, 1)
                # Score higher when closer to band_min (more urgent within window)
                position = 1.0 - (days - band_min) / window_size
                score += position * 60.0

            # --- Trend weight scales with horizon ---
            if trend == 'increasing':
                if horizon_int <= 7:
                    score += 10.0          # minor bonus for short windows
                elif horizon_int <= 30:
                    score += 30.0          # important for monthly planning
                else:
                    score += 55.0          # dominant factor for long-term windows
            elif trend == 'stable':
                score += 5.0

            # --- Overstock opportunity (relevant for longer windows) ---
            if status == 'overstock' and trend == 'increasing' and horizon_int >= 30:
                score += 20.0              # high stock + rising demand = opportunity

            # --- Risk amplifier ---
            if risk == 'HIGH':
                score += 15.0
            elif risk == 'MEDIUM':
                score += 7.0

            # --- Rating quality ---
            score += rating * 2.0

            return round(score, 2)

        def _build_product_row(p, a, score=0):
            img = _product_image_url(p.product_id, p.category or '', p.product_name or '')
            return {
                'product_id': p.product_id,
                'product_name': p.product_name,
                'brand': p.brand,
                'category': p.category or 'Other',
                'total_sales': score,       # repurposed as relevance score for this window
                'forecast_trend': (a.forecast_trend if a else 'stable') or 'stable',
                'stock_status': (a.stock_status if a else 'normal') or 'normal',
                'stockout_risk': (a.stockout_risk if a else 'LOW') or 'LOW',
                'avg_rating': round(float(a.avg_rating or 0), 1) if a else 0.0,
                'image_url': img,
                'days_until_stockout': round(float(a.days_until_stockout), 1) if (a and a.days_until_stockout is not None) else None,
                'forecast_horizon': horizon_int,
                'priority_score': a.priority_score if a else 0.0,
            }

        # --- Pull candidates: products in the horizon band + rising-trend products ---
        candidate_rows = db.session.query(Product, AnalysisResult).join(
            AnalysisResult,
            (AnalysisResult.product_id == Product.product_id) &
            (AnalysisResult.dealer_id == Product.dealer_id)
        ).filter(
            Product.dealer_id == dealer_id,
        ).filter(
            # Either falls in the band OR is a rising-trend product relevant to longer windows
            db.or_(
                db.and_(
                    AnalysisResult.days_until_stockout >= band_min,
                    AnalysisResult.days_until_stockout <= band_max,
                ),
                db.and_(
                    AnalysisResult.forecast_trend == 'increasing',
                    AnalysisResult.days_until_stockout > 0,
                ) if horizon_int >= 21 else db.and_(
                    AnalysisResult.days_until_stockout >= 0,
                    AnalysisResult.days_until_stockout <= horizon_int,
                ),
            )
        ).limit(200).all()

        # Score and sort candidates
        scored = []
        for p, a in candidate_rows:
            s = _horizon_score(a, horizon_int)
            if s > 0:
                scored.append((s, p, a))
        scored.sort(key=lambda x: x[0], reverse=True)

        # Apply category diversity: max 2 per category
        top_products = []
        cat_count = {}
        for s, p, a in scored:
            cat = (p.category or 'Other').strip()
            if cat_count.get(cat, 0) < 2:
                top_products.append(_build_product_row(p, a, score=round(s, 1)))
                cat_count[cat] = cat_count.get(cat, 0) + 1
            if len(top_products) >= 10:
                break

        # Fill any remaining slots with diverse top-rated products
        if len(top_products) < 10:
            existing_ids = {r['product_id'] for r in top_products}
            fill_rows = db.session.query(Product, AnalysisResult).outerjoin(
                AnalysisResult,
                (AnalysisResult.product_id == Product.product_id) &
                (AnalysisResult.dealer_id == Product.dealer_id)
            ).filter(
                Product.dealer_id == dealer_id,
                ~Product.product_id.in_(existing_ids)
            ).order_by(_desc(AnalysisResult.avg_rating)).limit(100).all()
            for p, a in fill_rows:
                cat = (p.category or 'Other').strip()
                if cat_count.get(cat, 0) < 2:
                    top_products.append(_build_product_row(p, a, score=0))
                    cat_count[cat] = cat_count.get(cat, 0) + 1
                if len(top_products) >= 10:
                    break

    except Exception as e:
        logger.error("Top products query error: %s", e)
        top_products = []

    # 2. Get Alerts and Health
    alert_feed = build_alert_feed(dealer_id, limit=10)
    category_health = compute_category_health_live(dealer_id)

    # 3. Final Response Construction
    # Fetch live counts if cache is missing or zero (to ensure we show "real" data from DB)
    actual_total = Product.query.filter_by(dealer_id=dealer_id).count()
    actual_under = AnalysisResult.query.filter_by(dealer_id=dealer_id, stock_status='understock').count()
    actual_over = AnalysisResult.query.filter_by(dealer_id=dealer_id, stock_status='overstock').count()
    actual_harmful = AnalysisResult.query.filter(
        AnalysisResult.dealer_id == dealer_id,
        AnalysisResult.risk_level.in_(['high', 'medium'])
    ).count()

    return jsonify({
        'total_products': actual_total,
        'understock_count': actual_under,
        'overstock_count': actual_over,
        'harmful_count': actual_harmful,
        'critical_alerts': sum(1 for a in alert_feed if a.get('severity') == 'critical'),
        'pipeline_status': cache.pipeline_status if cache else 'done',
        'pipeline_running': (cache.pipeline_status in ('running', 'pending')) if cache else False,
        'top_products': top_products,
        'alert_feed': alert_feed,
        'category_health': category_health,
        'last_updated': cache.last_updated.isoformat() + 'Z' if (cache and cache.last_updated) else datetime.utcnow().isoformat() + 'Z',
    })


@dashboard_bp.route('/alerts', methods=['GET'])
@require_auth
def get_alerts():
    """GET /api/dashboard/alerts — returns all unread notifications grouped by type."""
    dealer_id = request.dealer_id
    unread_only = request.args.get('unread', 'false').lower() == 'true'
    alerts = get_dealer_notifications(dealer_id, unread_only=unread_only)
    grouped = {}
    for a in alerts:
        t = a['type']
        grouped.setdefault(t, []).append(a)
    return jsonify({'alerts': alerts, 'grouped': grouped, 'total': len(alerts)})


@dashboard_bp.route('/alerts/<int:notif_id>/read', methods=['POST'])
@require_auth
def mark_alert_read(notif_id):
    """POST /api/dashboard/alerts/{id}/read — mark notification as read."""
    dealer_id = request.dealer_id
    success = mark_notification_read(notif_id, dealer_id)
    if success:
        return jsonify({'success': True, 'notif_id': notif_id})
    return jsonify({'error': 'Notification not found'}), 404


@dashboard_bp.route('/pipeline-status', methods=['GET'])
@require_auth
def get_pipeline_status():
    """GET /api/dashboard/pipeline-status — polling endpoint for frontend."""
    dealer_id = request.dealer_id
    progress = get_pipeline_progress(dealer_id)

    if progress['status'] == 'idle':
        cache = DashboardCache.query.filter_by(dealer_id=dealer_id).first()
        if cache:
            status = cache.pipeline_status or 'pending'

            # Auto-reset a stuck pipeline: running >10 min with 0 progress
            if status == 'running' and cache.pipeline_started_at:
                elapsed = (datetime.utcnow() - cache.pipeline_started_at).total_seconds()
                if elapsed > 600 and (cache.pipeline_progress or 0) < 5:
                    logger.warning("Auto-resetting stuck pipeline for dealer %s (elapsed %.0fs)", dealer_id, elapsed)
                    cache.pipeline_status = 'done'
                    cache.pipeline_progress = 100
                    db.session.commit()
                    status = 'done'

            return jsonify({
                'status': status,
                'progress_pct': cache.pipeline_progress or (100 if status == 'done' else 0),
                'products_processed': cache.total_products if status == 'done' else 0,
                'total_products': cache.total_products or 0,
                'started_at': cache.pipeline_started_at.isoformat() + 'Z' if cache.pipeline_started_at else None,
                'last_updated': cache.last_updated.isoformat() + 'Z' if cache.last_updated else None,
            })
        return jsonify({'status': 'done', 'progress_pct': 100, 'products_processed': 0, 'total_products': 0})

    return jsonify(progress)
@dashboard_bp.route('/notifications/<int:notif_id>/read', methods=['POST'])
@require_auth
def mark_notif_read(notif_id):
    """POST /api/dashboard/notifications/<id>/read — mark a specific alert as read."""
    from services.notification_service import mark_notification_read
    success = mark_notification_read(notif_id, request.dealer_id)
    if success:
        return jsonify({'status': 'success'})
    return jsonify({'status': 'error', 'message': 'Notification not found'}), 404

@dashboard_bp.route('/notifications/read-all', methods=['POST'])
@require_auth
def mark_all_notifs_read():
    """POST /api/dashboard/notifications/read-all — clear all unread alerts."""
    try:
        Notification.query.filter_by(dealer_id=request.dealer_id, is_read=False).update({'is_read': True})
        db.session.commit()
        return jsonify({'status': 'success'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500
