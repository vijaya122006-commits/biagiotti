# routes/upload_routes.py
from flask import Blueprint, request
from controllers.upload_controller import handle_upload
from utils.auth import require_auth   # Bug 3 fix: import auth decorator

upload_bp = Blueprint('upload', __name__)

@upload_bp.route('-sales', methods=['POST'])
@require_auth                          # Bug 3 fix: enforce authentication
def upload_sales():
    return handle_upload(request)
