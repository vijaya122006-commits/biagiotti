"""
backend/create_demo_db.py
=========================
Creates demo_store.sqlite with realistic Indian cosmetic demo data.
Run once during Render build: python create_demo_db.py
"""
import json
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

# ── ensure backend dir is in path so imports work ────────────────────────────
_BACKEND_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_BACKEND_DIR))

os.environ.setdefault('SECRET_KEY', 'biagiotti-demo-secret-2026')
os.environ.setdefault('FLASK_DEBUG', 'false')

from flask import Flask
from database.models import (
    db, Dealer, Product, Inventory, Sale, Review,
    AnalysisResult, HarmfulChemical, Notification, DashboardCache
)

DB_PATH = _BACKEND_DIR / 'demo_store.sqlite'

# ── Flask mini-app just for DB init ──────────────────────────────────────────
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'biagiotti-demo-secret-2026'
db.init_app(app)

# ─────────────────────────────────────────────────────────────────────────────
# PRODUCT CATALOGUE  (25 products)
# ─────────────────────────────────────────────────────────────────────────────
PRODUCTS = [
    # id, name, brand, category, price, cost, ingredients, skin_suitability, safety_score
    ("P001", "Radiance Glow Serum", "Biagiotti Glow", "Serum", 1299, 520,
     "Niacinamide, Hyaluronic Acid, Vitamin C, Glycerin, Aloe Vera",
     "All Skin Types", 92.0),
    ("P002", "Intense Moisture Cream", "Biagiotti Luxe", "Moisturizer", 899, 360,
     "Shea Butter, Ceramides, Peptides, Squalane, Rose Water",
     "Dry, Normal", 95.0),
    ("P003", "Matte Velvet Foundation", "Biagiotti Color", "Makeup", 1499, 600,
     "Titanium Dioxide, Zinc Oxide, Dimethicone, Talc, Mica",
     "Oily, Combination", 78.0),
    ("P004", "Deep Pore Cleanser", "Deep Clean", "Cleanser", 499, 180,
     "Salicylic Acid, Tea Tree Oil, Witch Hazel, Aloe Vera, Panthenol",
     "Oily, Acne-Prone", 88.0),
    ("P005", "SPF 50+ Invisible Sunscreen", "Safe Shield", "Suncare", 799, 320,
     "Avobenzone, Octinoxate, Zinc Oxide, Aloe Vera, Vitamin E",
     "All Skin Types", 83.0),
    ("P006", "Retinol Night Repair", "Biagiotti Science", "Treatment", 1899, 760,
     "Retinol 0.3%, Peptides, Ceramides, Niacinamide, Hyaluronic Acid",
     "Mature, Normal", 87.0),
    ("P007", "Rose Petal Toner", "Petal Fresh", "Toner", 599, 220,
     "Rose Water, Glycerin, Witch Hazel, Hyaluronic Acid, Allantoin",
     "All Skin Types", 96.0),
    ("P008", "Charcoal Detox Mask", "Clear Skin", "Mask", 699, 280,
     "Activated Charcoal, Kaolin Clay, Tea Tree Oil, Bentonite, Aloe Vera",
     "Oily, Combination", 89.0),
    ("P009", "Caffeine Eye Rescue Serum", "Biagiotti Science", "Eye Care", 1199, 480,
     "Caffeine 5%, Peptides, Hyaluronic Acid, Vitamin K, Aloe Vera",
     "All Skin Types", 91.0),
    ("P010", "Keratin Repair Shampoo", "Silky Strands", "Hair Care", 649, 260,
     "Keratin Protein, Argan Oil, Biotin, Panthenol, Glycerin",
     "Damaged Hair", 90.0),
    ("P011", "Plump & Shine Lip Gloss", "Biagiotti Color", "Lip Care", 449, 160,
     "Castor Oil, Vitamin E, Shea Butter, Beeswax, Peppermint Oil",
     "All Lip Types", 94.0),
    ("P012", "Brightening Body Lotion", "Glow Lab", "Body Care", 749, 300,
     "Kojic Acid, Vitamin C, Niacinamide, Shea Butter, Squalane",
     "All Skin Types", 86.0),
    ("P013", "Hydra-Boost Serum", "Pure Essence", "Serum", 1099, 440,
     "Hyaluronic Acid 2%, Ceramides, Panthenol, Aloe Vera, Glycerin",
     "Dry, Dehydrated", 97.0),
    ("P014", "Silk Touch Moisturizer", "Biagiotti Luxe", "Moisturizer", 999, 400,
     "Collagen Peptides, Jojoba Oil, Squalane, Vitamin E, Rose Hip",
     "Normal, Combination", 93.0),
    ("P015", "HD Concealer Palette", "Biagiotti Color", "Makeup", 1299, 520,
     "Titanium Dioxide, Talc, Mica, Dimethicone, Cyclopentasiloxane",
     "All Skin Types", 80.0),
    ("P016", "Gentle Foam Cleanser", "Pure Essence", "Cleanser", 549, 200,
     "Glycerin, Aloe Vera, Chamomile Extract, Panthenol, Allantoin",
     "Sensitive, Dry", 98.0),
    ("P017", "Tinted Sunscreen SPF 40", "Safe Shield", "Suncare", 899, 360,
     "Zinc Oxide, Iron Oxides, Niacinamide, Glycerin, Dimethicone",
     "All Skin Types", 85.0),
    ("P018", "Vitamin C Brightening Serum", "Glow Lab", "Serum", 1399, 560,
     "Ascorbic Acid 15%, Ferulic Acid, Vitamin E, Hyaluronic Acid, Niacinamide",
     "Dull, Hyperpigmented", 88.0),
    ("P019", "AHA BHA Exfoliating Toner", "Biagiotti Science", "Toner", 799, 320,
     "Glycolic Acid 5%, Salicylic Acid 2%, Lactic Acid, Aloe Vera, Panthenol",
     "Oily, Acne-Prone", 82.0),
    ("P020", "Overnight Glow Mask", "Biagiotti Glow", "Mask", 1199, 480,
     "Bakuchiol, Niacinamide, Hyaluronic Acid, Ceramides, Vitamin C",
     "All Skin Types", 94.0),
    ("P021", "Peptide Eye Cream", "Biagiotti Luxe", "Eye Care", 1499, 600,
     "Matrixyl 3000, Argireline, Hyaluronic Acid, Vitamin K, Caffeine",
     "Mature, All Types", 92.0),
    ("P022", "Argan Oil Hair Mask", "Silky Strands", "Hair Care", 799, 320,
     "Argan Oil, Keratin, Biotin, Vitamin E, Panthenol",
     "Dry, Damaged Hair", 95.0),
    ("P023", "Matte Liquid Lipstick", "Biagiotti Color", "Lip Care", 649, 260,
     "Isododecane, Dimethicone, Vitamin E, Jojoba Oil, Beeswax",
     "All Lip Types", 85.0),
    ("P024", "Firming Body Butter", "Pure Essence", "Body Care", 899, 360,
     "Shea Butter, Cocoa Butter, Collagen, Vitamin E, Almond Oil",
     "All Skin Types", 96.0),
    ("P025", "Salicylic Acid Spot Treatment", "Clear Skin", "Treatment", 499, 200,
     "Salicylic Acid 2%, Tea Tree Oil, Niacinamide, Zinc, Aloe Vera",
     "Oily, Acne-Prone", 86.0),
]

REVIEW_TEXTS = {
    "positive": [
        "Absolutely love this product! My skin feels amazing after just 2 weeks of use.",
        "Best purchase I've made this year. The texture is perfect and it absorbs quickly.",
        "My customers keep coming back for this one. Excellent results every time.",
        "Outstanding quality. Very effective and the fragrance is subtle and pleasant.",
        "I have recommended this to all my regular customers. They all love it!",
        "The formula is incredible. Noticeable difference within the first week.",
        "Premium feel at an affordable price. Will definitely stock up more.",
        "Great for Indian skin tones. Works perfectly in our humid climate.",
    ],
    "neutral": [
        "Decent product, does what it promises. Nothing extraordinary but reliable.",
        "Average performance. Works okay for some customers, not all.",
        "Good packaging but the results take time to show. Patience needed.",
        "Fair pricing for what you get. Middle of the road product.",
        "Some customers love it, some are indifferent. Mixed reactions overall.",
    ],
    "negative": [
        "Not the best for very oily skin types. A bit heavy for summer use.",
        "A few customers reported mild irritation. May not suit very sensitive skin.",
        "The pump dispenser can be inconsistent. Packaging needs improvement.",
        "Slightly disappointed with the quantity for the price. Expected more.",
    ],
}

REVIEWER_NAMES = [
    "Priya Sharma", "Anjali Singh", "Deepika Nair", "Sunita Patel", "Kavitha Reddy",
    "Meena Gupta", "Ritu Agarwal", "Pooja Menon", "Ananya Bose", "Shalini Iyer",
    "Divya Pillai", "Rekha Choudhary", "Nandini Joshi", "Lalitha Kumar", "Sneha Das",
    "Preethi Verma", "Archana Shah", "Swathi Rao", "Bhavana Mishra", "Geeta Khanna",
]

CITIES = ["Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai", "Pune", "Kolkata", "Ahmedabad"]
REGIONS = ["Maharashtra", "Delhi NCR", "Karnataka", "Telangana", "Tamil Nadu", "Gujarat", "West Bengal"]
EVENT_TYPES = ["Regular Sale", "Festival Offer", "Clearance", "New Launch", "Reorder", "Bulk Order"]

HARMFUL_CHEMICALS = [
    ("Parabens (Methylparaben, Propylparaben)", "4418-26-2", "HIGH",
     "Endocrine disruptor, linked to breast cancer risk", "Restricted in EU cosmetics",
     "Preservative", "Paraben", 8, "EWG Database"),
    ("Sodium Lauryl Sulfate (SLS)", "151-21-3", "MEDIUM",
     "Skin irritant, may cause contact dermatitis", "Permitted with concentration limits",
     "Surfactant", "Sulfate", 5, "FDA Database"),
    ("Formaldehyde", "50-00-0", "HIGH",
     "Known human carcinogen, causes allergic reactions", "Banned in EU, restricted globally",
     "Preservative", "Aldehyde", 9, "IARC List"),
    ("Hydroquinone", "123-31-9", "HIGH",
     "Potential carcinogen, causes ochronosis with prolonged use", "Banned in EU, OTC restricted in US",
     "Skin Lightener", "Phenol", 8, "WHO Report"),
    ("Phthalates (DBP, DEHP)", "84-74-2", "HIGH",
     "Endocrine disruption, developmental toxicity", "Banned in EU cosmetics",
     "Plasticizer", "Phthalate", 8, "EWG Database"),
]


def make_demo_db():
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"[INFO] Removed old {DB_PATH.name}")

    with app.app_context():
        db.create_all()
        print("[INFO] Tables created.")

        # ── Dealers ───────────────────────────────────────────────────────────
        d1 = Dealer(name="Demo Dealer", email="demo@cosmetic.ai",
                    shop_name="Biagiotti Demo Store", city="Mumbai", phone="9876543210")
        d1.set_password("demo1234")

        d2 = Dealer(name="Test Biagiotti", email="test@biagiotti.com",
                    shop_name="Biagiotti Test Shop", city="Bangalore", phone="9123456780")
        d2.set_password("test1234")

        db.session.add_all([d1, d2])
        db.session.flush()
        dealer_ids = [d1.dealer_id, d2.dealer_id]
        print(f"[INFO] Created 2 dealers: IDs {dealer_ids}")

        # ── Harmful Chemicals (global reference data) ─────────────────────────
        for cname, cas, risk, health, legal, pcat, scat, sev, src in HARMFUL_CHEMICALS:
            db.session.add(HarmfulChemical(
                chemical_name=cname, cas_number=cas, risk_level=risk,
                health_risk=health, legal_status=legal, primary_category=pcat,
                sub_category=scat, severity_score=sev, source_dataset=src,
            ))
        print("[INFO] Added 5 harmful chemicals.")

        db.session.flush()

        # ── Products + Inventory + Sales + Reviews + Analysis for each dealer ─
        for dealer_id in dealer_ids:
            rng = random.Random(dealer_id * 42)  # deterministic per dealer

            top_products_list = []
            alert_feed_list = []
            category_counts = {}
            total_harmful = 0
            total_understock = 0
            total_overstock = 0
            critical_alerts = 0

            for idx, (pid, pname, brand, cat, price, cost, ingr, skin, safety) in enumerate(PRODUCTS):
                # ── Product ───────────────────────────────────────────────────
                p = Product(
                    product_id=pid, dealer_id=dealer_id,
                    product_name=pname, brand=brand, category=cat,
                    price=float(price), cost_price=float(cost),
                    ingredients=ingr, skin_suitability=skin,
                    country="India", label="Retail", is_verified=True,
                )
                db.session.add(p)
                category_counts[cat] = category_counts.get(cat, 0) + 1

                # ── Inventory ─────────────────────────────────────────────────
                # Make ~20% of products low stock for alerts
                is_low = idx % 5 == 0
                stock = rng.randint(5, 25) if is_low else rng.randint(60, 200)
                reorder = 50
                inv = Inventory(
                    product_id=pid, dealer_id=dealer_id,
                    current_stock=float(stock), reorder_level=float(reorder),
                    lead_time_days=14.0,
                    last_restocked=datetime.utcnow() - timedelta(days=rng.randint(10, 45)),
                )
                db.session.add(inv)

                if stock < reorder:
                    total_understock += 1
                    critical_alerts += 1
                    alert_feed_list.append({
                        "type": "low_stock", "product": pname,
                        "message": f"Only {stock} units left. Reorder recommended.",
                        "severity": "high" if stock < 15 else "medium"
                    })

                if stock > 150:
                    total_overstock += 1

                # ── Sales — last 6 months (Dec 2025 – May 2026) ───────────────
                base_units = rng.randint(30, 120)
                months = [
                    (2025, 12), (2026, 1), (2026, 2),
                    (2026, 3), (2026, 4), (2026, 5),
                ]
                monthly_sales = []
                for yr, mo in months:
                    trend_factor = 1.0 + (months.index((yr, mo)) * 0.04)
                    units = max(5, int(base_units * trend_factor * rng.uniform(0.8, 1.2)))
                    revenue = units * price * rng.uniform(0.9, 1.0)
                    sale_date = datetime(yr, mo, rng.randint(1, 28))
                    db.session.add(Sale(
                        product_id=pid, dealer_id=dealer_id,
                        brand=brand,
                        region=rng.choice(REGIONS), city=rng.choice(CITIES),
                        event_type=rng.choice(EVENT_TYPES),
                        year=yr, month=mo,
                        units_sold=float(units), revenue=round(revenue, 2),
                        sell_through_pct=round(rng.uniform(0.55, 0.95), 2),
                        avg_daily_footfall=float(rng.randint(40, 200)),
                        sale_date=sale_date,
                    ))
                    monthly_sales.append(units)

                # ── Reviews (3-5 per product) ──────────────────────────────────
                n_reviews = rng.randint(3, 5)
                ratings = []
                sentiments = []
                for _ in range(n_reviews):
                    rating = rng.choices([4, 5, 3, 2], weights=[40, 35, 15, 10])[0]
                    if rating >= 4:
                        txt = rng.choice(REVIEW_TEXTS["positive"])
                        slabel = "positive"; sscore = rng.uniform(0.55, 0.95)
                    elif rating == 3:
                        txt = rng.choice(REVIEW_TEXTS["neutral"])
                        slabel = "neutral"; sscore = rng.uniform(-0.1, 0.2)
                    else:
                        txt = rng.choice(REVIEW_TEXTS["negative"])
                        slabel = "negative"; sscore = rng.uniform(-0.6, -0.15)

                    rdate = datetime.utcnow() - timedelta(days=rng.randint(5, 180))
                    db.session.add(Review(
                        product_id=pid, dealer_id=dealer_id,
                        source="demo", platform=rng.choice(["Nykaa", "Amazon", "Flipkart", "In-Store"]),
                        rating=float(rating),
                        review_title=txt[:60],
                        review_body=txt,
                        reviewer_name=rng.choice(REVIEWER_NAMES),
                        skin_type_mentioned=rng.choice(["Oily", "Dry", "Combination", "Normal", "Sensitive", ""]),
                        verified_purchase=rng.random() > 0.3,
                        helpful_votes=rng.randint(0, 45),
                        sentiment_score=round(sscore, 3),
                        sentiment_label=slabel,
                        is_synthetic=True,
                        review_date=rdate,
                    ))
                    ratings.append(rating)
                    sentiments.append(sscore)

                avg_rating = round(sum(ratings) / len(ratings), 2)
                avg_sentiment = round(sum(sentiments) / len(sentiments), 3)

                # ── Analysis Result ────────────────────────────────────────────
                forecast_vals = [int(monthly_sales[-1] * (1 + 0.05 * i)) for i in range(1, 7)]
                trend = "up" if forecast_vals[-1] > monthly_sales[-1] else "stable"
                stock_status = "Low Stock" if stock < reorder else ("Overstocked" if stock > 150 else "Healthy")
                days_out = round(stock / max(1, monthly_sales[-1] / 30), 1)
                risk = "HIGH" if days_out < 14 else ("MEDIUM" if days_out < 30 else "LOW")
                safe_status = "Safe" if safety >= 85 else ("Moderate" if safety >= 70 else "Unsafe")

                if safety < 70:
                    total_harmful += 1

                harmful_found = []
                if "Paraben" in ingr or "paraben" in ingr.lower():
                    harmful_found.append({"name": "Paraben compound", "risk": "HIGH"})

                recs = [
                    {"action": "Restock immediately", "priority": "HIGH"} if stock < reorder else
                    {"action": "Maintain current stock levels", "priority": "LOW"},
                    {"action": f"Feature in {rng.choice(['Nykaa', 'Instagram', 'WhatsApp'])} campaign", "priority": "MEDIUM"},
                ]

                db.session.add(AnalysisResult(
                    product_id=pid, dealer_id=dealer_id,
                    demand_forecast_json=json.dumps({"forecast": forecast_vals, "months": 6}),
                    forecast_trend=trend,
                    stock_status=stock_status,
                    days_until_stockout=days_out,
                    stockout_risk=risk,
                    skin_type_detected=skin,
                    skin_confidence=round(rng.uniform(0.75, 0.98), 2),
                    harmful_ingredients_json=json.dumps(harmful_found),
                    safety_score=safety,
                    safety_status=safe_status,
                    risk_level=risk,
                    stock_decision="RESTOCK" if stock < reorder else "HOLD",
                    decision_reason="Stock below reorder level" if stock < reorder else "Stock levels adequate",
                    priority_score=round(rng.uniform(0.5, 0.99), 2),
                    recommendations_json=json.dumps(recs),
                    avg_rating=avg_rating,
                    review_count=n_reviews,
                    sentiment_avg=avg_sentiment,
                    verification_status="verified",
                    analyzed_at=datetime.utcnow(),
                ))

                top_products_list.append({
                    "product_id": pid, "product_name": pname,
                    "brand": brand, "category": cat,
                    "avg_rating": avg_rating, "safety_score": safety,
                    "stock": stock, "trend": trend,
                })

            # sort top products by rating
            top_products_list.sort(key=lambda x: x["avg_rating"], reverse=True)

            # ── Notifications ──────────────────────────────────────────────────
            db.session.add(Notification(
                dealer_id=dealer_id, product_id="P001",
                notif_type="low_stock", severity="high",
                title="⚠️ Low Stock Alert: Radiance Glow Serum",
                message="Radiance Glow Serum has only 12 units left. Reorder level is 50. Place order immediately.",
                is_read=False,
            ))
            db.session.add(Notification(
                dealer_id=dealer_id, product_id="P003",
                notif_type="safety_alert", severity="medium",
                title="🔬 Safety Review: Matte Velvet Foundation",
                message="Titanium Dioxide in Matte Velvet Foundation requires regulatory review for nano-particle compliance.",
                is_read=False,
            ))
            db.session.add(Notification(
                dealer_id=dealer_id, product_id="P010",
                notif_type="restock_reminder", severity="low",
                title="📦 Restock Reminder: Keratin Repair Shampoo",
                message="Keratin Repair Shampoo is selling well. Consider placing a bulk order for the upcoming festival season.",
                is_read=True,
            ))

            # ── Dashboard Cache ────────────────────────────────────────────────
            cat_health = [
                {"category": cat, "count": cnt,
                 "health": "Good" if cnt >= 2 else "Fair"}
                for cat, cnt in category_counts.items()
            ]

            db.session.add(DashboardCache(
                dealer_id=dealer_id,
                total_products=len(PRODUCTS),
                understock_count=total_understock,
                overstock_count=total_overstock,
                harmful_count=total_harmful,
                critical_alerts=critical_alerts,
                top_products_json=json.dumps(top_products_list[:10]),
                alert_feed_json=json.dumps(alert_feed_list[:10]),
                category_health_json=json.dumps(cat_health),
                pipeline_status="done",
                pipeline_progress=100,
                pipeline_started_at=datetime.utcnow() - timedelta(minutes=5),
                last_updated=datetime.utcnow(),
                next_scheduled=datetime.utcnow() + timedelta(hours=24),
            ))

            print(f"[INFO] Dealer {dealer_id}: {len(PRODUCTS)} products, "
                  f"{total_understock} low-stock, {critical_alerts} alerts.")

        db.session.commit()
        print(f"\n✅ demo_store.sqlite created at {DB_PATH}")
        print("   Login: demo@cosmetic.ai / demo1234")
        print("   Login: test@biagiotti.com / test1234")


if __name__ == '__main__':
    make_demo_db()
