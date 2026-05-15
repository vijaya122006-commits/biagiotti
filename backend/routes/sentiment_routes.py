# routes/sentiment_routes.py
from flask import Blueprint
from controllers.sentiment_controller import get_skin_analysis

sentiment_bp = Blueprint('sentiment', __name__)

@sentiment_bp.route('/<product_name>', methods=['GET'])
def analyze_skin_type(product_name):
    return get_skin_analysis(product_name)
