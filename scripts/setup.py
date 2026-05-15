"""
backend/setup.py — One-command setup script for Cosmetic Intelligence System

Run this once:
    python backend/setup.py

It will:
1. Create all database tables
2. Import all CSV data into the database
3. Create a demo dealer account
4. Generate synthetic data for the demo account
5. Run the initial ML pipeline
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app import app
from database.db import init_db
from database.csv_importer import import_all_csvs
from database.synthetic_generator import generate_synthetic_data_for_dealer
from database.models import db, Dealer
from services.analysis_engine import run_full_pipeline_for_dealer


def setup():
    print("=" * 60)
    print("  Cosmetic Intelligence — One-Time Setup")
    print("=" * 60)

    with app.app_context():
        # Step 1: Create tables
        print("\n[1/5] Creating database tables...")
        db.create_all()
        print("      ✅ Tables created")

        # Step 2: Import CSVs
        print("\n[2/5] Importing CSV datasets...")
        try:
            import_all_csvs()
            print("      ✅ CSV data imported")
        except Exception as e:
            print(f"      ⚠️  CSV import had errors (non-fatal): {e}")

        # Step 3: Create demo dealer
        print("\n[3/5] Creating demo dealer account...")
        demo = Dealer.query.filter_by(email='demo@cosmetic.ai').first()
        if not demo:
            demo = Dealer(
                name='Demo Dealer',
                email='demo@cosmetic.ai',
                shop_name='Cosmetic Intelligence Demo',
                city='Hyderabad',
                is_sandbox=True,
            )
            demo.set_password('demo123')
            db.session.add(demo)
            db.session.commit()
            print(f"      ✅ Demo account created: demo@cosmetic.ai / demo123")
        else:
            print(f"      ✅ Demo account exists: demo@cosmetic.ai / demo123")

        # Step 4: Generate synthetic data
        print("\n[4/5] Generating synthetic inventory and reviews...")
        existing_products = db.session.query(db.func.count()).filter(
            db.text(f"dealer_id = {demo.dealer_id}")
        ).scalar() if False else None

        try:
            from database.models import Product
            count = Product.query.filter_by(dealer_id=demo.dealer_id).count()
            if count < 100:
                result = generate_synthetic_data_for_dealer(demo.dealer_id)
                print(f"      ✅ Generated {result.get('products_generated', 0)} products, "
                      f"{result.get('reviews_generated', 0)} reviews")
            else:
                print(f"      ✅ Synthetic data already exists ({count} products)")
        except Exception as e:
            print(f"      ⚠️  Synthetic data generation had errors: {e}")

        # Step 5: Run ML pipeline
        print("\n[5/5] Running initial ML analysis pipeline...")
        print("      (This may take 1-3 minutes for 500 products...)")
        try:
            result = run_full_pipeline_for_dealer(demo.dealer_id)
            print(f"      ✅ Pipeline complete — {result.get('products_analyzed', 0)} products analyzed")
            print(f"         Understock: {result.get('understock', 0)}")
            print(f"         Overstock:  {result.get('overstock', 0)}")
            print(f"         Harmful:    {result.get('harmful', 0)}")
        except Exception as e:
            print(f"      ⚠️  Pipeline had errors: {e}")
            import traceback
            traceback.print_exc()

        print("\n" + "=" * 60)
        print("  Setup Complete!")
        print("  Start the server: python backend/app.py")
        print("  Open:             http://localhost:5050")
        print("  Login:            demo@cosmetic.ai / demo123")
        print("=" * 60)


if __name__ == '__main__':
    setup()
