# routes/similarity_routes.py
from flask import Blueprint
from controllers.similarity_controller import get_similar_products, get_safe_alternatives

similarity_bp = Blueprint('similarity', __name__)

@similarity_bp.route('/<product_name>', methods=['GET'])
def get_similarity(product_name):
    return get_similar_products(product_name)

@similarity_bp.route('/alternatives/<product_name>', methods=['GET']) 
def get_alternatives(product_name):
    return get_safe_alternatives(product_name)
