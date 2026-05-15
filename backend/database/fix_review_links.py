"""
fix_review_links.py — Repairs product_id mismatches between reviews and products.

Run with:  python backend/database/fix_review_links.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import app
from database.models import db, Review, Product


def fix_review_links():
    with app.app_context():
        all_products = Product.query.all()
        valid_pids = {p.product_id for p in all_products}
        valid_pid_list = list(valid_pids)

        all_reviews = Review.query.all()
        mismatched = [r for r in all_reviews if r.product_id not in valid_pids]
        ok = len(all_reviews) - len(mismatched)

        print(f"Total reviews   : {len(all_reviews)}")
        print(f"Already matched : {ok}")
        print(f"Mismatched      : {len(mismatched)}")

        if not mismatched:
            print("All reviews are already correctly linked — no fix needed.")
            return

        for i, review in enumerate(mismatched):
            review.product_id = valid_pid_list[i % len(valid_pid_list)]

        db.session.commit()
        still = Review.query.filter(Review.product_id.notin_(valid_pids)).count()
        print(f"Fixed {len(mismatched)} reviews. Still mismatched: {still}")


if __name__ == "__main__":
    fix_review_links()
