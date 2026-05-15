# utils/response_builder.py
from flask import jsonify

def build_success_response(data=None, message="Success", status_code=200):
    response = {
        "status": "success",
        "message": message
    }
    if data is not None:
        response["data"] = data
    return jsonify(response), status_code

def build_error_response(message="An error occurred", status_code=400):
    response = {
        "status": "error",
        "message": message
    }
    return jsonify(response), status_code
