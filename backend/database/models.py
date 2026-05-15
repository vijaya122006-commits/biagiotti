"""
database/models.py — Complete SQLAlchemy models for the Cosmetic Intelligence System
"""
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash

db = SQLAlchemy()


class Dealer(db.Model):
    __tablename__ = 'dealers'
    dealer_id       = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name            = db.Column(db.String(200), nullable=False)
    email           = db.Column(db.String(200), unique=True, nullable=False)
    password_hash   = db.Column(db.String(500), nullable=False)
    shop_name       = db.Column(db.String(200))
    city            = db.Column(db.String(100))
    phone           = db.Column(db.String(20))
    is_sandbox      = db.Column(db.Boolean, default=False)
    reset_token     = db.Column(db.String(100), unique=True, nullable=True)
    reset_token_expiry = db.Column(db.DateTime, nullable=True)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)
    last_login      = db.Column(db.DateTime)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


class Product(db.Model):
    __tablename__ = 'products'
    id              = db.Column(db.Integer, primary_key=True, autoincrement=True)
    product_id      = db.Column(db.String(50), nullable=False)
    dealer_id       = db.Column(db.Integer, db.ForeignKey('dealers.dealer_id'), nullable=False)
    product_name    = db.Column(db.String(500), nullable=False)
    brand           = db.Column(db.String(200))
    brand_id        = db.Column(db.String(50))
    category        = db.Column(db.String(100))
    price           = db.Column(db.Float)
    cost_price      = db.Column(db.Float)
    ingredients     = db.Column(db.Text)
    skin_suitability = db.Column(db.String(200))
    monk_category   = db.Column(db.Float)
    shade           = db.Column(db.String(100))
    hex_color       = db.Column(db.String(10))
    country         = db.Column(db.String(100))
    label           = db.Column(db.String(100))
    is_verified     = db.Column(db.Boolean, default=False)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at      = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint('product_id', 'dealer_id', name='uq_product_dealer'),
    )


class Inventory(db.Model):
    __tablename__ = 'inventory'
    inventory_id        = db.Column(db.Integer, primary_key=True, autoincrement=True)
    product_id          = db.Column(db.String(50), nullable=False)
    dealer_id           = db.Column(db.Integer, db.ForeignKey('dealers.dealer_id'), nullable=False)
    current_stock       = db.Column(db.Float, default=0)
    reorder_level       = db.Column(db.Float, default=50)
    lead_time_days      = db.Column(db.Float, default=14)
    last_restocked      = db.Column(db.DateTime)
    updated_at          = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Sale(db.Model):
    __tablename__ = 'sales'
    sale_id         = db.Column(db.Integer, primary_key=True, autoincrement=True)
    product_id      = db.Column(db.String(50), nullable=False)
    dealer_id       = db.Column(db.Integer, db.ForeignKey('dealers.dealer_id'), nullable=False)
    brand           = db.Column(db.String(200))
    region          = db.Column(db.String(100))
    city            = db.Column(db.String(100))
    event_type      = db.Column(db.String(100))
    year            = db.Column(db.Integer)
    month           = db.Column(db.Integer)
    units_sold      = db.Column(db.Float, default=0)
    revenue         = db.Column(db.Float, default=0)
    sell_through_pct = db.Column(db.Float)
    avg_daily_footfall = db.Column(db.Float)
    sale_date       = db.Column(db.DateTime)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)


class Review(db.Model):
    __tablename__ = 'reviews'
    review_id           = db.Column(db.Integer, primary_key=True, autoincrement=True)
    product_id          = db.Column(db.String(50), nullable=False)
    dealer_id           = db.Column(db.Integer, db.ForeignKey('dealers.dealer_id'), nullable=False)
    source              = db.Column(db.String(50), default='synthetic')
    platform            = db.Column(db.String(50))
    rating              = db.Column(db.Float)
    review_title        = db.Column(db.String(500))
    review_body         = db.Column(db.Text)
    reviewer_name       = db.Column(db.String(200))
    skin_type_mentioned = db.Column(db.String(100))
    verified_purchase   = db.Column(db.Boolean, default=False)
    helpful_votes       = db.Column(db.Integer, default=0)
    sentiment_score     = db.Column(db.Float)
    sentiment_label     = db.Column(db.String(20))
    is_synthetic        = db.Column(db.Boolean, default=True)
    review_date         = db.Column(db.DateTime)
    created_at          = db.Column(db.DateTime, default=datetime.utcnow)


class AnalysisResult(db.Model):
    __tablename__ = 'analysis_results'
    result_id               = db.Column(db.Integer, primary_key=True, autoincrement=True)
    product_id              = db.Column(db.String(50), nullable=False)
    dealer_id               = db.Column(db.Integer, db.ForeignKey('dealers.dealer_id'), nullable=False)
    demand_forecast_json    = db.Column(db.Text)
    forecast_trend          = db.Column(db.String(20))
    stock_status            = db.Column(db.String(20))
    days_until_stockout     = db.Column(db.Float)
    stockout_risk           = db.Column(db.String(10))
    skin_type_detected      = db.Column(db.String(200))
    skin_confidence         = db.Column(db.Float)
    harmful_ingredients_json = db.Column(db.Text)
    safety_score            = db.Column(db.Float)
    safety_status           = db.Column(db.String(20))
    risk_level              = db.Column(db.String(10))
    stock_decision          = db.Column(db.String(100))
    decision_reason         = db.Column(db.Text)
    priority_score          = db.Column(db.Float)
    recommendations_json    = db.Column(db.Text)
    avg_rating              = db.Column(db.Float)
    review_count            = db.Column(db.Integer, default=0)
    sentiment_avg           = db.Column(db.Float)
    verification_status     = db.Column(db.String(30))
    analyzed_at             = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint('product_id', 'dealer_id', name='uq_analysis_product_dealer'),
    )


class HarmfulChemical(db.Model):
    __tablename__ = 'harmful_chemicals'
    chemical_id         = db.Column(db.Integer, primary_key=True, autoincrement=True)
    chemical_name       = db.Column(db.String(300), nullable=False)
    cas_number          = db.Column(db.String(50))
    risk_level          = db.Column(db.String(10))
    health_risk         = db.Column(db.Text)
    legal_status        = db.Column(db.String(200))
    primary_category    = db.Column(db.String(100))
    sub_category        = db.Column(db.String(100))
    severity_score      = db.Column(db.Integer, default=5)
    source_dataset      = db.Column(db.String(100))
    added_at            = db.Column(db.DateTime, default=datetime.utcnow)


class Notification(db.Model):
    __tablename__ = 'notifications'
    notif_id        = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dealer_id       = db.Column(db.Integer, db.ForeignKey('dealers.dealer_id'), nullable=False)
    product_id      = db.Column(db.String(50))
    notif_type      = db.Column(db.String(30))
    title           = db.Column(db.String(300))
    message         = db.Column(db.Text)
    severity        = db.Column(db.String(10), default='medium')
    is_read         = db.Column(db.Boolean, default=False)
    created_at      = db.Column(db.DateTime, default=datetime.utcnow)


class DashboardCache(db.Model):
    __tablename__ = 'dashboard_cache'
    cache_id            = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dealer_id           = db.Column(db.Integer, db.ForeignKey('dealers.dealer_id'), unique=True)
    total_products      = db.Column(db.Integer, default=0)
    understock_count    = db.Column(db.Integer, default=0)
    overstock_count     = db.Column(db.Integer, default=0)
    harmful_count       = db.Column(db.Integer, default=0)
    critical_alerts     = db.Column(db.Integer, default=0)
    top_products_json   = db.Column(db.Text)
    alert_feed_json     = db.Column(db.Text)
    category_health_json = db.Column(db.Text)
    pipeline_status     = db.Column(db.String(20), default='pending')
    pipeline_progress   = db.Column(db.Integer, default=0)
    pipeline_started_at = db.Column(db.DateTime)
    last_updated        = db.Column(db.DateTime)
    next_scheduled      = db.Column(db.DateTime)


class DealerDatabaseConnection(db.Model):
    """Stores a dealer's external database connection so we can sync it in real-time."""
    __tablename__ = 'dealer_db_connections'

    conn_id             = db.Column(db.Integer, primary_key=True, autoincrement=True)
    dealer_id           = db.Column(db.Integer, db.ForeignKey('dealers.dealer_id'), unique=True, nullable=False)

    # Connection details
    db_type             = db.Column(db.String(20), nullable=False)   # postgresql | mysql | sqlite
    host                = db.Column(db.String(255))
    port                = db.Column(db.Integer)
    db_name             = db.Column(db.String(255))
    username            = db.Column(db.String(255))
    password_encrypted  = db.Column(db.Text)   # base64-encoded (not production-grade — use vault in prod)

    # Table & column mapping (JSON strings)
    products_table      = db.Column(db.String(255), default='products')
    sales_table         = db.Column(db.String(255), default='sales')
    column_map_json     = db.Column(db.Text)   # {"product_id": "id", "product_name": "name", ...}

    # Sync settings
    sync_interval_min   = db.Column(db.Integer, default=30)          # how often to pull (minutes)
    is_active           = db.Column(db.Boolean, default=True)

    # Sync state
    sync_status         = db.Column(db.String(20), default='idle')   # idle|running|done|error
    last_synced_at      = db.Column(db.DateTime)
    last_sync_rows      = db.Column(db.Integer, default=0)
    last_error          = db.Column(db.Text)

    created_at          = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at          = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

