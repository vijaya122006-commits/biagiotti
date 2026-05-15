"""
routes/review_routes.py — Reviews per product
"""
import logging
from datetime import datetime
from flask import Blueprint, jsonify, request
from database.models import db, Review
from utils.auth import require_auth

review_routes_bp = Blueprint('review_routes', __name__)
logger = logging.getLogger("review_routes")


@review_routes_bp.route('/products/<product_id>/reviews', methods=['GET'])
@require_auth
def get_reviews(product_id):
    """GET /api/products/{product_id}/reviews — paginated review list."""
    dealer_id = request.dealer_id
    source = request.args.get('source')
    rating_min = request.args.get('rating_min', type=float)
    rating_max = request.args.get('rating_max', type=float)
    sentiment = request.args.get('sentiment')
    page = int(request.args.get('page', 1))
    per_page = int(request.args.get('per_page', 20))

    q = Review.query.filter_by(product_id=product_id, dealer_id=dealer_id)
    if source:
        q = q.filter_by(source=source)
    if rating_min is not None:
        q = q.filter(Review.rating >= rating_min)
    if rating_max is not None:
        q = q.filter(Review.rating <= rating_max)
    if sentiment:
        q = q.filter_by(sentiment_label=sentiment)

    total = q.count()
    reviews = q.order_by(Review.review_date.desc()).offset((page - 1) * per_page).limit(per_page).all()

    # Fallback: same category, this dealer only (never cross-dealer)
    if not reviews:
        from database.models import Product
        product = Product.query.filter_by(product_id=product_id, dealer_id=dealer_id).first()
        if product and product.category:
            cat_pids = [
                p.product_id for p in
                Product.query.filter_by(category=product.category, dealer_id=dealer_id).limit(20).all()
            ]
            reviews = Review.query.filter(
                Review.product_id.in_(cat_pids),
                Review.dealer_id == dealer_id
            ).order_by(Review.review_date.desc()).limit(per_page).all()
            total = len(reviews)

    return jsonify({
        'reviews': [{
            'review_id': r.review_id,
            'reviewer_name': r.reviewer_name,
            'rating': r.rating,
            'review_title': r.review_title,
            'review_body': r.review_body,
            'skin_type_mentioned': r.skin_type_mentioned,
            'source': r.source,
            'platform': r.platform,
            'sentiment_score': r.sentiment_score,
            'sentiment_label': r.sentiment_label,
            'verified_purchase': r.verified_purchase,
            'helpful_votes': r.helpful_votes,
            'is_synthetic': r.is_synthetic,
            'review_date': r.review_date.isoformat() + 'Z' if r.review_date else None,
        } for r in reviews],
        'total': total,
        'page': page,
        'per_page': per_page,
    })


@review_routes_bp.route('/products/<product_id>/reviews', methods=['POST'])
@require_auth
def add_review(product_id):
    """POST /api/products/{product_id}/reviews — dealer manually adds a review."""
    dealer_id = request.dealer_id
    data = request.get_json(silent=True) or {}

    review_body = (data.get('review_body') or data.get('text') or '').strip()
    if not review_body:
        return jsonify({'error': 'review_body is required'}), 400

    # Run sentiment analysis
    try:
        from services.ml_service import analyze_sentiment
        sent = analyze_sentiment(review_body)
        sent_score = sent.get('score', 0.0)
        sent_label = sent.get('sentiment', 'neutral')
    except Exception:
        sent_score = 0.0
        sent_label = 'neutral'

    try:
        review = Review(
            product_id=product_id,
            dealer_id=dealer_id,
            source='manual',
            platform=data.get('platform', 'manual'),
            rating=float(data.get('rating')) if data.get('rating') else None,
            review_title=data.get('review_title'),
            review_body=review_body,
            reviewer_name=data.get('reviewer_name'),
            skin_type_mentioned=data.get('skin_type_mentioned'),
            verified_purchase=bool(data.get('verified_purchase', False)),
            helpful_votes=0,
            sentiment_score=sent_score,
            sentiment_label=sent_label,
            is_synthetic=False,
            review_date=datetime.utcnow(),
        )
        db.session.add(review)
        db.session.commit()

        return jsonify({
            'success': True,
            'review_id': review.review_id,
            'sentiment': sent_label,
            'sentiment_score': sent_score,
        }), 201

    except Exception as e:
        db.session.rollback()
        logger.error("Add review failed: %s", e)
        return jsonify({'error': 'Failed to add review'}), 500


@review_routes_bp.route('/reviews/summary/<product_id>', methods=['GET'])
@require_auth
def get_review_summary(product_id):
    """GET /api/reviews/summary/{product_id} — aggregated review stats."""
    dealer_id = request.dealer_id
    reviews = Review.query.filter_by(product_id=product_id, dealer_id=dealer_id).all()

    if not reviews:
        return jsonify({
            'product_id': product_id,
            'avg_rating': 0,
            'review_count': 0,
            'rating_distribution': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            'sentiment': {'positive': 0, 'neutral': 0, 'negative': 0},
            'skin_type_breakdown': {},
        })

    import numpy as np
    ratings = [r.rating for r in reviews if r.rating]
    avg_rating = float(np.mean(ratings)) if ratings else 0

    rating_dist = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    for rat in ratings:
        k = round(rat)
        if k in rating_dist:
            rating_dist[k] += 1

    sentiments = {'positive': 0, 'neutral': 0, 'negative': 0}
    for r in reviews:
        if r.sentiment_label in sentiments:
            sentiments[r.sentiment_label] += 1

    skin_types = {}
    for r in reviews:
        if r.skin_type_mentioned:
            skin_types[r.skin_type_mentioned] = skin_types.get(r.skin_type_mentioned, 0) + 1

    return jsonify({
        'product_id': product_id,
        'avg_rating': round(avg_rating, 2),
        'review_count': len(reviews),
        'rating_distribution': rating_dist,
        'sentiment': sentiments,
        'skin_type_breakdown': skin_types,
    })
