# routes/report_routes.py
from flask import Blueprint
from controllers.report_controller import get_dashboard_summary, generate_report

report_bp = Blueprint('report', __name__)

@report_bp.route('/summary', methods=['GET'])
def summary():
    return get_dashboard_summary()

@report_bp.route('/download-report', methods=['GET'])
def download():
    return generate_report()
