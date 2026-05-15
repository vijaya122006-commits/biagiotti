"""fix_prices.py — Run once: python backend/database/fix_prices.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app
from database.models import db, Product

USD_TO_INR = 83.5

def fix_prices():
    with app.app_context():
        products = Product.query.filter(
            Product.price != None,
            Product.price < 200
        ).all()
        fixed = 0
        for p in products:
            if p.price and p.price < 200:
                old = p.price
                p.price = round(p.price * USD_TO_INR, 0)
                print(f"  {p.product_id}: ₹{old} → ₹{p.price}")
                fixed += 1
        db.session.commit()
        print(f"\nFixed {fixed} product prices (USD → INR @ {USD_TO_INR})")

if __name__ == '__main__':
    fix_prices()
