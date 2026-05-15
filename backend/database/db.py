"""
database/db.py — Database initialization helper
"""
from database.models import db


def init_db(app):
    """Initialize Flask-SQLAlchemy with the app and create all tables."""
    db.init_app(app)
    with app.app_context():
        db.create_all()
        print("Database tables created.")
