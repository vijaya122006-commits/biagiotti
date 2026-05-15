# routes/safety_routes.py
from flask import Blueprint
from controllers.safety_controller import check_safety

safety_bp = Blueprint('safety', __name__)

@safety_bp.route('/<product_name>', methods=['GET'])
def analyze_safety(product_name):
    return check_safety(product_name)
