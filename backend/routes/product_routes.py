import os
import json
import logging
import threading
import hashlib as _hashlib
from datetime import datetime
from flask import Blueprint, jsonify, request
from database.models import db, Product, Inventory, AnalysisResult, Sale, Review
from utils.auth import require_auth
import numpy as np

product_routes_bp = Blueprint('product_routes', __name__)
logger = logging.getLogger("product_routes")

# AI-generated category images — served via Flask static
_API_BASE = os.environ.get('API_BASE_URL', 'http://localhost:5050').rstrip('/')

def _static(filename: str) -> str:
    return f'{_API_BASE}/static/categories/{filename}'

# ── Step 1: Name-keyword → image  (ordered, most-specific first) ─────────────
# This table is checked against the PRODUCT NAME because DB categories like
# "Moisturizer" are far too coarse — serums, oils, and foundations are all
# bucketed into "Moisturizer" in the Sephora dataset.
_NAME_KEYWORD_MAP = [
    # SPF / sun care (check before "cream" / "lotion" to catch "SPF 50 Moisturizer")
    (['spf', 'sunscreen', 'sun screen', 'uv protect', 'solar', 'broad spectrum'],
     _static('sun_protect.png')),
    # Foundation / BB / CC / concealer
    (['foundation', 'bb cream', 'cc cream', 'cc+', ' bb ', 'concealer', 'tinted moisturizer',
      'tinted hydrating', 'skin tint', 'cushion compact', 'powder foundation',
      'liquid foundation', 'stick foundation'],
     _static('foundation.png')),
    # Face mask (before serum/toner)
    (['clay mask', 'sheet mask', 'sleeping mask', 'overnight mask', 'mud mask',
      'peel off', 'gel mask', 'eye mask', 'lip mask', 'charcoal mask', 'face mask'],
     _static('face_mask.png')),
    # Eye area
    (['eye cream', 'eye gel', 'eye serum', 'eye mask', 'under eye', 'dark circle',
      'eye contour', 'eye lift'],
     _static('eye_cream.png')),
    # Lip
    (['lip balm', 'lip butter', 'lip oil', 'lip gloss', 'lip care', 'lip treatment',
      'lip serum', 'lip mask'],
     _static('lip_care.png')),
    # Hair
    (['shampoo', 'conditioner', 'hair mask', 'hair serum', 'hair oil', 'hair care',
      'hair treatment', 'scalp', 'dry shampoo'],
     _static('hair_care.png')),
    # Cleanser / face wash
    (['cleanser', 'face wash', 'foaming wash', 'micellar', 'cleansing oil',
      'cleansing balm', 'makeup remover', 'cleansing milk', 'facial wash',
      'anti-pollution', 'mousse cleanser', 'gel cleanser'],
     _static('cleanser.png')),
    # Toner / essence / mist
    (['toner', 'tonic', 'essence', 'facial mist', 'face mist', 'rose water',
      'treatment essence', 'first treatment', 'lotion toner', 'clarifying toner',
      'exfoliating toner'],
     _static('toner.png')),
    # Serum / concentrate / ampoule
    (['serum', 'ampoule', 'concentrate', 'booster', 'retinol', 'vitamin c',
      'hyaluronic acid', 'niacinamide', 'peptide', 'acid serum', 'facial serum',
      'night serum', 'resurfacing serum', 'brightening serum', 'firming serum'],
     _static('serum.png')),
    # Face oil (after serum to not catch "oil-free")
    (['facial oil', 'face oil', 'rosehip oil', 'marula oil', 'jojoba oil',
      'argan oil', 'sleeping oil', 'recovery oil', 'night oil', 'luxury oil'],
     _static('serum.png')),   # oil shown as serum bottle (amber dropper)
    # Treatment / exfoliator
    (['treatment', 'exfoliat', 'peel', 'aha', 'bha', 'pha', 'glycolic', 'lactic',
      'salicylic', 'resurfac', 'renewal', 'refining'],
     _static('treatment.png')),
    # Moisturizer / cream / lotion (broad — must come last)
    (['moisturizer', 'moisturiser', 'cream', 'lotion', 'hydrator', 'gel cream',
      'water cream', 'emulsion', 'balm', 'butter moistur', 'day cream', 'night cream'],
     _static('moisturizer.png')),
]

# ── Step 2: Category-string → image  (for well-categorised uploaded products) ─
_CATEGORY_IMAGE_MAP = {
    'moisturizer':   _static('moisturizer.png'),
    'serum':         _static('serum.png'),
    'sunscreen':     _static('sun_protect.png'),
    'sun screen':    _static('sun_protect.png'),
    'sun protect':   _static('sun_protect.png'),
    'spf':           _static('sun_protect.png'),
    'foundation':    _static('foundation.png'),
    'cleanser':      _static('cleanser.png'),
    'face wash':     _static('cleanser.png'),
    'face cleanser': _static('cleanser.png'),
    'toner':         _static('toner.png'),
    'eye cream':     _static('eye_cream.png'),
    'eye gel':       _static('eye_cream.png'),
    'lip care':      _static('lip_care.png'),
    'lip balm':      _static('lip_care.png'),
    'lip gloss':     _static('lip_care.png'),
    'mask':          _static('face_mask.png'),
    'face mask':     _static('face_mask.png'),
    'sheet mask':    _static('face_mask.png'),
    'hair care':     _static('hair_care.png'),
    'shampoo':       _static('hair_care.png'),
    'conditioner':   _static('hair_care.png'),
    'hair serum':    _static('hair_care.png'),
    'treatment':     _static('treatment.png'),
    '__default__':   _static('moisturizer.png'),
}


def _product_image_url(product_id: str, category: str = '', product_name: str = '') -> str:
    """
    Three-step image resolution:
      1. Product NAME keywords  — most reliable (DB categories are too coarse)
      2. Category string match  — for dealer-uploaded products with clean categories
      3. Static default         — always a real cosmetic image, never Unsplash
    """
    name_lower = (product_name or '').strip().lower()
    cat_lower  = (category or '').strip().lower()

    # ── Step 1: product name keyword scan ────────────────────────────────────
    if name_lower:
        for keywords, img_url in _NAME_KEYWORD_MAP:
            if any(kw in name_lower for kw in keywords):
                return img_url

    # ── Step 2: category exact match ─────────────────────────────────────────
    if cat_lower in _CATEGORY_IMAGE_MAP:
        return _CATEGORY_IMAGE_MAP[cat_lower]

    # ── Step 2b: category partial match ──────────────────────────────────────
    for key, url in _CATEGORY_IMAGE_MAP.items():
        if key != '__default__' and key in cat_lower:
            return url

    # ── Step 3: safe static fallback ─────────────────────────────────────────
    return _CATEGORY_IMAGE_MAP['__default__']




@product_routes_bp.route('', methods=['GET'])
@require_auth
def get_products():
    """GET /api/products — paginated product list with filters."""
    dealer_id = request.dealer_id
    page = int(request.args.get('page', 1))
    per_page = min(int(request.args.get('per_page', 20)), 5000)
    category = request.args.get('category')
    stock_status = request.args.get('stock_status')
    risk_level = request.args.get('risk_level')
    search = request.args.get('search', '').strip()

    q = db.session.query(Product, AnalysisResult, Inventory).outerjoin(
        AnalysisResult, (AnalysisResult.product_id == Product.product_id) &
                        (AnalysisResult.dealer_id == Product.dealer_id)
    ).outerjoin(
        Inventory, (Inventory.product_id == Product.product_id) &
                   (Inventory.dealer_id == Product.dealer_id)
    ).filter(Product.dealer_id == dealer_id)

    if category:
        q = q.filter(Product.category == category)
    if search:
        q = q.filter(Product.product_name.ilike(f'%{search}%'))
    if stock_status:
        statuses = [s.strip() for s in stock_status.split(',')]
        q = q.filter(AnalysisResult.stock_status.in_(statuses))
    if risk_level:
        levels = [r.strip() for r in risk_level.split(',')]
        q = q.filter(AnalysisResult.risk_level.in_(levels))

    total = q.count()
    rows = q.offset((page - 1) * per_page).limit(per_page).all()

    products = []
    for product, analysis, inventory in rows:
        products.append({
            'product_id': product.product_id,
            'product_name': product.product_name,
            'brand': product.brand,
            'category': product.category,
            'price': product.price,
            'skin_suitability': product.skin_suitability,
            'is_verified': product.is_verified,
            'image_url': _product_image_url(product.product_id, product.category or '', product.product_name or ''),
            'current_stock': inventory.current_stock if inventory else None,
            'reorder_level': inventory.reorder_level if inventory else None,
            # Analysis fields
            'stock_status': analysis.stock_status if analysis else None,
            'days_until_stockout': analysis.days_until_stockout if analysis else None,
            'stockout_risk': analysis.stockout_risk if analysis else None,
            'risk_level': analysis.risk_level if analysis else None,
            'safety_score': analysis.safety_score if analysis else None,
            'safety_status': analysis.safety_status if analysis else None,
            'avg_rating': analysis.avg_rating if analysis else None,
            'review_count': analysis.review_count if analysis else 0,
            'forecast_trend': analysis.forecast_trend if analysis else None,
            'stock_decision': analysis.stock_decision if analysis else None,
            'priority_score': analysis.priority_score if analysis else None,
            'verification_status': analysis.verification_status if analysis else None,
        })

    return jsonify({
        'products': products,
        'total': total,
        'page': page,
        'per_page': per_page,
        'pages': (total + per_page - 1) // per_page,
    })


@product_routes_bp.route('/<product_id>/analysis', methods=['GET'])
@require_auth
def get_product_analysis(product_id):
    """GET /api/products/{product_id}/analysis — full drill-down."""
    dealer_id = request.dealer_id

    product = Product.query.filter_by(product_id=product_id, dealer_id=dealer_id).first()
    if not product:
        return jsonify({'error': 'Product not found'}), 404

    analysis = AnalysisResult.query.filter_by(product_id=product_id, dealer_id=dealer_id).first()
    inventory = Inventory.query.filter_by(product_id=product_id, dealer_id=dealer_id).first()
    reviews = Review.query.filter_by(product_id=product_id, dealer_id=dealer_id).all()
    # Fallback: same category, this dealer only (never cross-dealer)
    if not reviews and product.category:
        cat_pids = [
            p.product_id for p in
            Product.query.filter_by(category=product.category, dealer_id=dealer_id).limit(20).all()
        ]
        reviews = Review.query.filter(
            Review.product_id.in_(cat_pids),
            Review.dealer_id == dealer_id
        ).limit(20).all()

    # Get sales history
    sales = Sale.query.filter_by(
        product_id=product_id, dealer_id=dealer_id
    ).order_by(Sale.year, Sale.month).all()
    sales_history = [{'year': s.year, 'month': s.month, 'units_sold': s.units_sold, 'revenue': s.revenue}
                     for s in sales]

    # Optimized aggregation: use SQL GROUP BY for rating distribution
    from sqlalchemy import func
    rating_stats = db.session.query(
        func.round(Review.rating).label('star'),
        func.count(Review.review_id).label('count')
    ).filter(
        Review.product_id == product_id,
        Review.dealer_id == dealer_id
    ).group_by(func.round(Review.rating)).all()
    
    rating_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for star, count in rating_stats:
        if star and int(star) in rating_dist:
            rating_dist[int(star)] = count

    # Optimized sentiment counts
    sentiment_stats = db.session.query(
        Review.sentiment_label,
        func.count(Review.review_id)
    ).filter(
        Review.product_id == product_id,
        Review.dealer_id == dealer_id
    ).group_by(Review.sentiment_label).all()
    
    total_reviews = sum(c for _, c in sentiment_stats)
    pos = next((c for s, c in sentiment_stats if s == 'positive'), 0)
    neg = next((c for s, c in sentiment_stats if s == 'negative'), 0)
    neu = total_reviews - pos - neg

    # Only fetch the 10 most recent reviews
    recent_reviews = Review.query.filter_by(
        product_id=product_id, dealer_id=dealer_id
    ).order_by(
        Review.review_date.desc()
    ).limit(10).all()

    # Parse harmful ingredients from DB (stored list)
    harmful_list = []
    if analysis and analysis.harmful_ingredients_json:
        try:
            harmful_list = json.loads(analysis.harmful_ingredients_json)
        except Exception:
            pass

    # Re-run harmful detector live to get safe_ingredients (DB stores only harmful list)
    safe_list = []
    live_harmful = harmful_list  # default to stored list
    try:
        from ml_service import svc as _svc
        if product.ingredients:
            det = _svc.detect_harmful(ingredient_text=product.ingredients, product_id=product_id)
            live_harmful = det.get('harmful_ingredients', harmful_list)
            safe_list    = det.get('safe_ingredients', [])
    except Exception:
        pass  # fall back to stored harmful list

    # Parse forecast — stored either as a list or as {"forecast":[...],"months":N}
    forecast = []
    if analysis and analysis.demand_forecast_json:
        try:
            raw = json.loads(analysis.demand_forecast_json)
            # Handle both formats: plain list OR {"forecast": [...], "months": N}
            if isinstance(raw, list):
                forecast = raw
            elif isinstance(raw, dict):
                forecast = raw.get('forecast', raw.get('values', []))
        except Exception:
            pass

    # Similar products are handled by a separate endpoint now
    similar_products = []

    return jsonify({
        'product': {
            'product_id': product.product_id,
            'product_name': product.product_name,
            'brand': product.brand,
            'category': product.category,
            'price': product.price,
            'cost_price': product.cost_price,
            'ingredients': product.ingredients,
            'skin_suitability': product.skin_suitability,
            'is_verified': product.is_verified,
            'image_url': _product_image_url(product.product_id, product.category or '', product.product_name or ''),
        },
        'inventory': {
            'current_stock': inventory.current_stock if inventory else 0,
            'reorder_level': inventory.reorder_level if inventory else 50,
            'lead_time_days': inventory.lead_time_days if inventory else 14,
            'last_restocked': inventory.last_restocked.isoformat() + 'Z' if inventory and inventory.last_restocked else None,
        },
        'forecast': {
            'values': forecast,
            'trend': analysis.forecast_trend if analysis else None,
            'stock_decision': analysis.stock_decision if analysis else None,
            'decision_reason': analysis.decision_reason if analysis else None,
            'days_until_stockout': analysis.days_until_stockout if analysis else None,
            'stockout_risk': analysis.stockout_risk if analysis else None,
            'priority_score': analysis.priority_score if analysis else None,
            'history': [s['units_sold'] for s in sales_history[-12:]],
        },
        'skin_analysis': {
            'skin_type': analysis.skin_type_detected if analysis else None,
            'confidence': analysis.skin_confidence if analysis else None,
        },
        'safety': {
            'safety_score': analysis.safety_score if analysis else None,
            'safety_status': analysis.safety_status if analysis else None,
            'risk_level': analysis.risk_level if analysis else None,
            'harmful_ingredients': live_harmful,
            'safe_ingredients': safe_list,
        },
        'reviews': {
            'avg_rating': analysis.avg_rating if analysis else 0,
            'review_count': total_reviews,
            'sentiment': {
                'positive_pct': round(pos / max(1, total_reviews) * 100),
                'neutral_pct': round(neu / max(1, total_reviews) * 100),
                'negative_pct': round(neg / max(1, total_reviews) * 100),
                'avg_score': analysis.sentiment_avg if analysis else 0,
            },
            'rating_distribution': rating_dist,
            'recent': [{
                'reviewer_name': r.reviewer_name,
                'rating': r.rating,
                'review_title': r.review_title,
                'review_body': r.review_body,
                'skin_type_mentioned': r.skin_type_mentioned,
                'sentiment_label': r.sentiment_label,
                'review_date': r.review_date.isoformat() + 'Z' if r.review_date else None,
                'is_synthetic': r.is_synthetic,
                'source': r.source,
            } for r in recent_reviews],
        },
        'similar_products': similar_products,
        'verification_status': analysis.verification_status if analysis else 'unverified',
        'sales_history': sales_history,
    })


@product_routes_bp.route('/<product_id>/recommendations', methods=['GET'])
@require_auth
def get_smart_recommendations(product_id):
    """GET /api/products/{product_id}/recommendations — trending alternatives with reasoning."""
    dealer_id = request.dealer_id
    top_n = int(request.args.get('top_n', 5))

    target = Product.query.filter_by(product_id=product_id, dealer_id=dealer_id).first()
    if not target:
        return jsonify({'recommendations': [], 'error': 'Product not found'}), 404

    target_analysis = AnalysisResult.query.filter_by(product_id=product_id, dealer_id=dealer_id).first()

    # Target avg monthly sales
    from sqlalchemy import func, nullslast
    target_sales = db.session.query(
        func.avg(Sale.units_sold)
    ).filter_by(product_id=product_id, dealer_id=dealer_id).scalar() or 0
    target_sales = float(target_sales)

    # 1. Start with the Similarity Engine to find "Proper" alternatives
    from ml_service import svc
    sim_res = svc.get_similar_products(product_id=product_id, top_n=20)
    sim_pids = [r['product_id'] for r in sim_res.get('results', [])]

    # 2. Get Analysis and Product data for these similar IDs
    candidates = []
    if sim_pids:
        candidates = db.session.query(Product, AnalysisResult).outerjoin(
            AnalysisResult, (AnalysisResult.product_id == Product.product_id) & (AnalysisResult.dealer_id == Product.dealer_id)
        ).filter(
            Product.dealer_id == dealer_id,
            Product.product_id.in_(sim_pids),
            # Exclude high-risk in the query so we get a usable pool
            db.or_(AnalysisResult.risk_level != 'high', AnalysisResult.risk_level.is_(None))
        ).all()

    # 3. Always complement with category-based safe performers.
    #    Filter risk_level != 'high' IN the SQL so the top-N limit is applied
    #    to the safe pool (not consumed by high-risk products that get discarded).
    cat_safe = db.session.query(Product, AnalysisResult).outerjoin(
        AnalysisResult, (AnalysisResult.product_id == Product.product_id) & (AnalysisResult.dealer_id == Product.dealer_id)
    ).filter(
        Product.dealer_id == dealer_id,
        Product.product_id != product_id,
        Product.category == target.category,
        db.or_(AnalysisResult.risk_level != 'high', AnalysisResult.risk_level.is_(None))
    ).order_by(
        nullslast(AnalysisResult.priority_score.desc())
    ).limit(30).all()
    candidates.extend(cat_safe)

    # 4. Cross-category fallback — also safe only, broader pool
    if len(candidates) < 5:
        any_safe = db.session.query(Product, AnalysisResult).outerjoin(
            AnalysisResult, (AnalysisResult.product_id == Product.product_id) & (AnalysisResult.dealer_id == Product.dealer_id)
        ).filter(
            Product.dealer_id == dealer_id,
            Product.product_id != product_id,
            db.or_(AnalysisResult.risk_level != 'high', AnalysisResult.risk_level.is_(None))
        ).order_by(
            nullslast(AnalysisResult.priority_score.desc())
        ).limit(30).all()
        candidates.extend(any_safe)

    # 5. Last resort: include medium risk products if still nothing found
    if len(candidates) < 3:
        medium_ok = db.session.query(Product, AnalysisResult).outerjoin(
            AnalysisResult, (AnalysisResult.product_id == Product.product_id) & (AnalysisResult.dealer_id == Product.dealer_id)
        ).filter(
            Product.dealer_id == dealer_id,
            Product.product_id != product_id,
        ).order_by(
            nullslast(AnalysisResult.priority_score.desc())
        ).limit(30).all()
        candidates.extend(medium_ok)

    # Dedup candidates
    seen_pids = {product_id}
    unique_candidates = []
    for p, a in candidates:
        if p and p.product_id not in seen_pids:
            unique_candidates.append((p, a))
            seen_pids.add(p.product_id)

    # 5. Process and score
    pids = [p.product_id for p, _ in unique_candidates]
    sales_stats = db.session.query(
        Sale.product_id,
        func.avg(Sale.units_sold).label('avg_monthly'),
        func.sum(Sale.units_sold).label('total_sales')
    ).filter(
        Sale.dealer_id == dealer_id,
        Sale.product_id.in_(pids)
    ).group_by(Sale.product_id).all()

    sales_map = {s.product_id: (float(s.avg_monthly or 0), float(s.total_sales or 0)) for s in sales_stats}

    final_candidates = []
    for product, analysis in unique_candidates:
        avg_m, tot_s = sales_map.get(product.product_id, (0.0, 0.0))
        final_candidates.append((product, analysis, avg_m, tot_s))

    risk_rank = {'high': 3, 'medium': 2, 'low': 1, 'none': 0}
    target_risk = target_analysis.risk_level if target_analysis else 'none'
    # Safely convert target_rating — guard against None
    target_rating = float(target_analysis.avg_rating or 0) if target_analysis else 0.0

    scored = []
    for product, analysis, avg_monthly, total_sales in final_candidates:
        if not product:
            continue

        # CRITICAL SAFETY FILTER: Never recommend high-risk products
        cand_risk = analysis.risk_level if analysis else 'none'
        if cand_risk == 'high':
            continue

        avg_monthly = float(avg_monthly or 0)
        # Base score: every non-high-risk alternative starts with 5 so it always
        # passes the final gate even when there is no sales / analysis data yet.
        score = 5
        reasons = []
        why_better = []

        # Sales velocity
        if target_sales > 0 and avg_monthly > target_sales * 1.2:
            pct = round((avg_monthly - target_sales) / target_sales * 100)
            score += 40
            reasons.append(f'Selling {pct}% more units/month than current product')
            why_better.append(f'Higher demand ({round(avg_monthly)}/mo vs {round(target_sales)}/mo)')
        elif avg_monthly > 0 and avg_monthly > target_sales:
            score += 20
            reasons.append('Slightly higher sales velocity')

        # Trend
        if analysis and analysis.forecast_trend == 'increasing':
            score += 25
            reasons.append('Demand is trending upward this season')
            why_better.append('Rising trend — good time to stock more')

        # Safety
        if risk_rank.get(cand_risk, 0) < risk_rank.get(target_risk, 0):
            score += 45  # Significant boost for safety
            reasons.append('🛡️ Safer alternative (No harmful ingredients)')
            why_better.append('Avoids harmful ingredients found in current product')
        elif cand_risk == 'none' and target_risk == 'none':
            score += 5

        # Rating
        cand_rating = float(analysis.avg_rating or 0) if analysis else 0.0
        if cand_rating and target_rating and cand_rating > target_rating + 0.2:
            score += 25
            reasons.append(f'Superior satisfaction (⭐{cand_rating:.1f})')
            why_better.append(f'Better customer feedback than current (⭐{target_rating:.1f})')
        elif cand_rating >= 4.0:
            score += 10
            reasons.append('Highly rated by customers')

        # Stock available
        if analysis and analysis.stock_status == 'normal':
            score += 10
            reasons.append('Well stocked and available')

        # Same skin type
        if (target.skin_suitability and product.skin_suitability and
                any(s.strip() in product.skin_suitability
                    for s in target.skin_suitability.split(','))):
            score += 10
            reasons.append('Suitable for same skin types')

        # Confident Replacement Check
        is_confident = False
        if avg_monthly > target_sales * 1.1 and (analysis and analysis.forecast_trend == 'increasing'):
            # Safe comparison now that both are plain floats
            if cand_rating >= target_rating:
                is_confident = True
                score += 50
                reasons.append('✨ Confident Replacement: High demand + Rising trend')
                why_better.append('Top-performing alternative with better growth potential')

        # Same category boost
        if product.category and target.category and product.category == target.category:
            score += 5

        if not reasons:
            reasons.append('Alternative product in your inventory')

        # Sort reasons to put "Confident" ones first
        reasons.sort(key=lambda x: 'Confident' in x, reverse=True)

        scored.append({
            'product_id': product.product_id,
            'product_name': product.product_name,
            'brand': product.brand or '',
            'category': product.category or '',
            'price': product.price or 0,
            'skin_suitability': product.skin_suitability or '',
            'avg_rating': round(cand_rating, 1) if cand_rating else None,
            'avg_monthly_sales': round(avg_monthly),
            'forecast_trend': analysis.forecast_trend if analysis else 'stable',
            'stock_status': analysis.stock_status if analysis else 'normal',
            'safety_status': analysis.safety_status if analysis else 'Safe',
            'recommendation_score': score,
            'recommendation_reasons': reasons[:3],
            'why_better_than_current': why_better[:2],
            'is_trending': (analysis.forecast_trend == 'increasing') if analysis else False,
            'is_confident': is_confident
        })

    scored.sort(key=lambda x: x['recommendation_score'], reverse=True)
    return jsonify({'recommendations': scored[:top_n], 'total': len(scored)})


@product_routes_bp.route('', methods=['POST'])
@require_auth
def add_product():
    """POST /api/products — add a new product and trigger single-product analysis."""
    dealer_id = request.dealer_id
    data = request.get_json(silent=True) or {}

    product_name = (data.get('product_name') or '').strip()
    price = data.get('price')

    if not product_name:
        return jsonify({'error': 'product_name is required'}), 400

    try:
        import hashlib as _h
        pid_hash = int(_h.md5(f"{dealer_id}{product_name}{datetime.utcnow().isoformat()}".encode()).hexdigest(), 16) % 100000
        product_id = f"MAN_{pid_hash:05d}"

        product = Product(
            product_id=product_id,
            dealer_id=dealer_id,
            product_name=product_name,
            brand=data.get('brand'),
            category=data.get('category'),
            price=float(price) if price else None,
            cost_price=float(data.get('cost_price')) if data.get('cost_price') else None,
            ingredients=data.get('ingredients'),
            skin_suitability=data.get('skin_suitability'),
        )
        db.session.add(product)

        inventory = Inventory(
            product_id=product_id,
            dealer_id=dealer_id,
            current_stock=float(data.get('current_stock', 0)),
            reorder_level=50,
            lead_time_days=14,
        )
        db.session.add(inventory)
        db.session.commit()

        # Trigger analysis in background
        def _analyze():
            try:
                from app import app
                from services.analysis_engine import run_full_pipeline_for_dealer
                with app.app_context():
                    run_full_pipeline_for_dealer(dealer_id)
            except Exception as e:
                logger.error("Background analysis failed: %s", e)

        thread = threading.Thread(target=_analyze, daemon=True)
        thread.start()

        return jsonify({
            'success': True,
            'product_id': product_id,
            'message': 'Product added. Analysis running in background.',
        }), 201

    except Exception as e:
        db.session.rollback()
        logger.error("Add product failed: %s", e)
        return jsonify({'error': 'Failed to add product'}), 500


@product_routes_bp.route('/<product_id>/stock', methods=['PATCH'])
@require_auth
def update_stock(product_id):
    """PATCH /api/products/{product_id}/stock — update stock level."""
    dealer_id = request.dealer_id
    data = request.get_json(silent=True) or {}

    new_stock = data.get('current_stock')
    if new_stock is None:
        return jsonify({'error': 'current_stock is required'}), 400

    try:
        inventory = Inventory.query.filter_by(
            product_id=product_id, dealer_id=dealer_id
        ).first()

        if not inventory:
            inventory = Inventory(product_id=product_id, dealer_id=dealer_id)
            db.session.add(inventory)

        inventory.current_stock = float(new_stock)
        inventory.last_restocked = datetime.utcnow()
        db.session.commit()

        # Recalculate days_until_stockout
        from services.analysis_engine import _get_monthly_sales, _compute_stock_status
        from app import app
        with app.app_context():
            monthly = _get_monthly_sales(product_id, dealer_id)
            stock_info = _compute_stock_status(float(new_stock), monthly)

            # Update analysis result
            analysis = AnalysisResult.query.filter_by(
                product_id=product_id, dealer_id=dealer_id
            ).first()
            if analysis:
                analysis.days_until_stockout = stock_info['days_until_stockout']
                analysis.stockout_risk = stock_info['stockout_risk']
                analysis.stock_status = stock_info['stock_status']
                db.session.commit()

        return jsonify({
            'success': True,
            'product_id': product_id,
            'current_stock': float(new_stock),
            'days_until_stockout': stock_info.get('days_until_stockout'),
            'stockout_risk': stock_info.get('stockout_risk'),
        })

    except Exception as e:
        db.session.rollback()
        logger.error("Stock update failed: %s", e)
        return jsonify({'error': 'Failed to update stock'}), 500
