# routes/forecast_routes.py
from flask import Blueprint, request
from controllers.forecast_controller import create_forecast

forecast_bp = Blueprint('forecast', __name__)

@forecast_bp.route('', methods=['POST'])
def run_forecast():
    # Can accept form-data or json
    data = request.json if request.is_json else request.form
    return create_forecast(data)
