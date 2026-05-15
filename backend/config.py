import os
from pathlib import Path
from dotenv import load_dotenv

_BACKEND_DIR = Path(__file__).resolve().parent
load_dotenv(_BACKEND_DIR / '.env')

# ── Secret Key — never raises, always has a fallback ─────────────────────────
_default_secret = os.environ.get('SECRET_KEY', 'biagiotti-demo-secret-2026')


class Config:
    SECRET_KEY = _default_secret
    DEBUG = os.environ.get('FLASK_DEBUG', 'false').lower() in ('true', '1', 'yes')
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB

    # SQLite demo DB — no PostgreSQL needed for Render demo deployment
    DATABASE_URL = os.environ.get(
        'DATABASE_URL',
        f'sqlite:///{_BACKEND_DIR / "demo_store.sqlite"}'
    )
    SQLALCHEMY_DATABASE_URI = DATABASE_URL
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    SQLALCHEMY_ENGINE_OPTIONS = {
        'pool_pre_ping': True,
        'pool_recycle': 300,
    }

    UPLOAD_FOLDER = str(_BACKEND_DIR / 'data')
    CORS_HEADERS = ['Content-Type', 'Authorization', 'Accept']

    # ── Mail — gracefully disabled if credentials not set ────────────────────
    MAIL_SERVER = os.environ.get('MAIL_SERVER', 'smtp.gmail.com')
    MAIL_PORT = int(os.environ.get('MAIL_PORT', 587))
    MAIL_USE_TLS = True
    MAIL_USE_SSL = False
    MAIL_USERNAME = os.environ.get('MAIL_USERNAME')       # None if not set — OK
    MAIL_PASSWORD = os.environ.get('MAIL_PASSWORD')       # None if not set — OK
    MAIL_DEFAULT_SENDER = os.environ.get('MAIL_USERNAME', 'noreply@biagiotti.ai')
    MAIL_SUPPRESS_SEND = not bool(os.environ.get('MAIL_USERNAME'))  # suppress if no creds

    # ── Frontend URL ──────────────────────────────────────────────────────────
    FRONTEND_URL = os.environ.get('FRONTEND_URL', 'https://biagiotti-cosmetic.onrender.com')


config = Config()
