from flask import Blueprint, request
from controllers.product_intelligence_controller import (
    handle_product_upload, search_products, get_all_products, clear_db, analyze_stored_product
)
from services.ml_service import svc

product_intelligence_bp = Blueprint('product_intl', __name__)

@product_intelligence_bp.route('/upload', methods=['POST'])
def upload():
    return handle_product_upload(request)

@product_intelligence_bp.route('/search', methods=['GET'])
def search():
    q = request.args.get('q', '')
    return search_products(q)

@product_intelligence_bp.route('/all', methods=['GET'])
def all_prods():
    return get_all_products()

@product_intelligence_bp.route('/clear', methods=['POST'])
def clear():
    return clear_db()

@product_intelligence_bp.route('/analyze', methods=['POST'])
def analyze():
    # Use request.get_json() for the analyze endpoint
    data = request.get_json() or {}
    return analyze_stored_product(data, svc)
