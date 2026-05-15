"""
utils/auth.py — JWT token helpers and auth decorator
"""
import os
import jwt
from datetime import datetime, timedelta
from functools import wraps
from flask import request, jsonify

import warnings as _warnings

_DEFAULT_SECRET = 'cosmetic-intel-secret-2026'
SECRET = os.environ.get('SECRET_KEY', _DEFAULT_SECRET)

if SECRET == _DEFAULT_SECRET:
    _warnings.warn(
        "\n[SECURITY] SECRET_KEY is not set — using the hardcoded default.\n"
        "Set a strong SECRET_KEY environment variable before deploying to production.\n"
        "  export SECRET_KEY=\"$(python3 -c 'import secrets; print(secrets.token_hex(32))')\"\n",
        stacklevel=2
    )


def generate_token(dealer_id: int) -> str:
    """Generate a JWT token valid for 7 days."""
    payload = {
        'dealer_id': dealer_id,
        'exp': datetime.utcnow() + timedelta(days=7),
        'iat': datetime.utcnow()
    }
    return jwt.encode(payload, SECRET, algorithm='HS256')


def decode_token(token: str) -> dict:
    """Decode and verify a JWT token."""
    return jwt.decode(token, SECRET, algorithms=['HS256'])


def get_dealer_id_from_token(auth_header: str) -> int:
    """Extract dealer_id from 'Bearer <token>' string."""
    if not auth_header or not auth_header.startswith('Bearer '):
        return None
    token = auth_header.replace('Bearer ', '').strip()
    try:
        data = decode_token(token)
        return data.get('dealer_id')
    except:
        return None


def require_auth(f):
    """Decorator — verifies JWT token and injects dealer_id into request."""
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        token = auth_header.replace('Bearer ', '').strip()
        if not token:
            return jsonify({'error': 'No token provided'}), 401
        try:
            data = decode_token(token)
            request.dealer_id = data['dealer_id']
        except jwt.ExpiredSignatureError:
            return jsonify({'error': 'Token expired'}), 401
        except jwt.InvalidTokenError:
            return jsonify({'error': 'Invalid token'}), 401
        return f(*args, **kwargs)
    return decorated
