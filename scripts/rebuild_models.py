"""
rebuild_models.py — Rebuild all ML models from current DB data
Run: python3 backend/rebuild_models.py
"""
import sys, os, pickle, json, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

# Ensure backend/ is on path (rebuild_models.py lives in scripts/)
BACKEND = Path(__file__).resolve().parent.parent / "backend"
sys.path.insert(0, str(BACKEND))

print("=== Biagiotti Model Rebuilder ===\n")

# ── 1. Load DB products ────────────────────────────────────────────────────────
print("[1/5] Loading products from database...")
from app import app
from database.models import db, Product, Sale, AnalysisResult

with app.app_context():
    products = Product.query.filter(
        Product.ingredients.isnot(None),
        Product.ingredients != ''
    ).all()
    print(f"  Products with ingredients: {len(products)}")

    all_products = Product.query.all()
    print(f"  Total products: {len(all_products)}")

    # Load sales for forecast model
    sales_rows = Sale.query.all()
    print(f"  Total sales rows: {len(sales_rows)}")

    # Collect all data we need before closing context
    product_data = []
    for p in all_products:
        product_data.append({
            'product_id': p.product_id,
            'product_name': p.product_name or '',
            'ingredients': p.ingredients or '',
            'skin_suitability': p.skin_suitability or '',
            'category': p.category or '',
            'price': p.price or 500.0,
        })

    # Sales data keyed by product_id
    from collections import defaultdict
    sales_by_pid = defaultdict(list)
    for s in sales_rows:
        sales_by_pid[s.product_id].append({
            'year': s.year or 2024,
            'month': s.month or 1,
            'units_sold': s.units_sold or 0,
        })

MODELS_DIR = BACKEND / "models"
MODELS_DIR.mkdir(exist_ok=True)

# ── 2. Rebuild Skin Type Classifier ──────────────────────────────────────────
print("\n[2/5] Building skin type classifier...")

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score

# Build training data from skin_suitability labels + ingredients
X_skin, y_skin = [], []
SKIN_KEYWORDS = {
    'oily':      ['oily', 'sebum', 'shine', 'pore', 'grease', 'mattify', 'oil-control', 'non-comedogenic', 'zinc'],
    'dry':       ['dry', 'hydrat', 'moistur', 'hyaluronic', 'glycerin', 'ceramide', 'squalane', 'flak', 'tight', 'nourish'],
    'sensitive': ['sensitive', 'gentle', 'calming', 'soothing', 'fragrance-free', 'hypoallergenic', 'redness', 'irritat', 'aloe'],
    'acne':      ['acne', 'salicylic', 'benzoyl', 'pimple', 'breakout', 'blemish', 'spot', 'niacinamide', 'retinol', 'tea tree'],
    'normal':    ['balance', 'normal', 'all skin', 'everyday', 'daily', 'gentle', 'light'],
    'combination': ['combination', 't-zone', 'balance', 'mattify', 'hydrat'],
}

def infer_skin_type(ingredients_text, suitability_text):
    """Infer primary skin type from ingredients + suitability field."""
    text = (ingredients_text + ' ' + suitability_text).lower()
    scores = {}
    for stype, keywords in SKIN_KEYWORDS.items():
        score = sum(1 for k in keywords if k in text)
        scores[stype] = score

    # Also use explicit suitability field
    suit = suitability_text.lower()
    if 'oily' in suit: scores['oily'] = scores.get('oily', 0) + 3
    if 'dry' in suit:  scores['dry'] = scores.get('dry', 0) + 3
    if 'sensitive' in suit: scores['sensitive'] = scores.get('sensitive', 0) + 3
    if 'acne' in suit or 'acne-prone' in suit: scores['acne'] = scores.get('acne', 0) + 3
    if 'combination' in suit: scores['combination'] = scores.get('combination', 0) + 3
    if 'normal' in suit: scores['normal'] = scores.get('normal', 0) + 2

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 'normal'

for p in product_data:
    label = infer_skin_type(p['ingredients'], p['skin_suitability'])
    X_skin.append(p['ingredients'])
    y_skin.append(label)

print(f"  Training samples: {len(X_skin)}")
from collections import Counter
label_counts = Counter(y_skin)
print(f"  Label distribution: {dict(label_counts)}")

# Build TF-IDF + Logistic Regression classifier
vectorizer_v2 = TfidfVectorizer(
    ngram_range=(1, 2),
    max_features=5000,
    min_df=2,
    sublinear_tf=True,
    strip_accents='ascii'
)
X_mat = vectorizer_v2.fit_transform(X_skin)

# Filter to labels with >=5 samples
valid_labels = {l for l, c in label_counts.items() if c >= 5}
mask = [i for i, l in enumerate(y_skin) if l in valid_labels]
X_mat_f = X_mat[mask]
y_f = [y_skin[i] for i in mask]

from sklearn.linear_model import LogisticRegression
skin_model_v2 = LogisticRegression(
    max_iter=1000,
    C=1.0,
    class_weight='balanced',
    random_state=42,
    solver='lbfgs',
)
skin_model_v2.fit(X_mat_f, y_f)

# Quick accuracy check
scores = cross_val_score(skin_model_v2, X_mat_f, y_f, cv=3, scoring='accuracy')
print(f"  CV accuracy: {scores.mean():.1%} ± {scores.std():.1%}")

with open(MODELS_DIR / "skin_model_v2.pkl", "wb") as f:
    pickle.dump(skin_model_v2, f)
with open(MODELS_DIR / "vectorizer_v2.pkl", "wb") as f:
    pickle.dump(vectorizer_v2, f)
print("  ✅ Saved skin_model_v2.pkl + vectorizer_v2.pkl")

# ── 3. Test skin detection ─────────────────────────────────────────────────────
print("\n[3/5] Testing skin detection on sample ingredients...")
test_cases = [
    ("Salicylic Acid, Niacinamide, Tea Tree Oil, Benzoyl Peroxide", "acne"),
    ("Hyaluronic Acid, Glycerin, Ceramide, Squalane, Shea Butter", "dry"),
    ("Zinc Oxide, Niacinamide, Mattifying Silica, Oil-control", "oily"),
    ("Aloe Vera, Oat Extract, Centella Asiatica, Fragrance-free, Hypoallergenic", "sensitive"),
    ("Mineral Oil, Petrolatum, Glycerin, Lanolin", "dry"),
]
for ingredients, expected in test_cases:
    X_test = vectorizer_v2.transform([ingredients])
    pred = skin_model_v2.predict(X_test)[0]
    proba = skin_model_v2.predict_proba(X_test)[0]
    conf = proba.max()
    status = "✅" if pred == expected else "⚠️ "
    print(f"  {status} Expected:{expected:12s} Got:{pred:12s} conf:{conf:.0%}")

# ── 4. Rebuild Forecast Model ─────────────────────────────────────────────────
print("\n[4/5] Building forecast model from real sales data...")

from datetime import datetime

def get_category(name):
    n = name.lower()
    if any(k in n for k in ['spf', 'sunscreen', 'sun protect', 'uv']): return 'sunscreen'
    if any(k in n for k in ['serum', 'ampoule', 'concentrate']): return 'serum'
    if any(k in n for k in ['eye cream', 'eye gel', 'under eye']): return 'eye'
    if any(k in n for k in ['lip balm', 'lip butter', 'lip']): return 'lip'
    if any(k in n for k in ['cleanser', 'face wash', 'scrub', 'exfoliant']): return 'cleanser'
    if any(k in n for k in ['mask', 'clay', 'mud']): return 'mask'
    if any(k in n for k in ['face oil', 'rosehip', 'argan']): return 'oil'
    if any(k in n for k in ['toner', 'tonic', 'mist', 'essence']): return 'toner'
    if any(k in n for k in ['cream', 'moisturizer', 'lotion', 'balm']): return 'cream'
    return 'default'

CAT_LABELS = ['sunscreen','serum','eye','lip','cleanser','mask','oil','toner','cream','default']
cat_le = {c: i for i, c in enumerate(CAT_LABELS)}

# Build training rows from sales data
# For each product with >=6 months of sales, create lag features
train_rows = []
pid_le = {}

for i, p in enumerate(product_data):
    pid = p['product_id']
    pid_le[pid] = i % 1000  # encode product ID
    sales = sorted(sales_by_pid.get(pid, []), key=lambda s: (s['year'], s['month']))
    if len(sales) < 4:
        continue

    monthly = [s['units_sold'] for s in sales]
    cat = get_category(p['product_name'])
    price = p['price']

    for t in range(3, len(monthly)):
        if monthly[t] <= 0:
            continue
        row = {
            'lag_1': monthly[t-1],
            'lag_2': monthly[t-2],
            'lag_4': monthly[min(t-4, 0)] if t >= 4 else monthly[0],
            'lag_8': monthly[t-8] if t >= 8 else monthly[0],
            'rolling_mean_4': np.mean(monthly[max(0, t-4):t]),
            'rolling_std_4': np.std(monthly[max(0, t-4):t]) if t >= 4 else 0,
            'trend': monthly[t-1] - monthly[max(0, t-4)],
            'month': sales[t]['month'],
            'week_of_year': (sales[t]['month'] * 4),
            'quarter': (sales[t]['month'] - 1) // 3 + 1,
            'season': ((sales[t]['month'] - 1) // 3) % 4 + 1,
            'is_holiday_season': 1 if sales[t]['month'] in [10, 11, 12] else 0,
            'price_scaled': price / 1000.0,
            'product_avg_sales': np.mean(monthly[:t]),
            'product_hash': (hash(pid) % 1000) / 1000.0,
            'category_encoded': cat_le.get(cat, 0),
            'product_id_encoded': pid_le.get(pid, 0),
            'target': monthly[t],
        }
        train_rows.append(row)

print(f"  Training rows built: {len(train_rows)}")

if len(train_rows) < 50:
    print("  ⚠️  Too few real sales rows — generating synthetic training data")
    # Generate synthetic training data for model fitting
    from hashlib import md5
    for i in range(2000):
        seed = i
        rng = np.random.RandomState(seed)
        base = rng.uniform(50, 500)
        monthly = [max(1, base + rng.normal(0, base*0.2)) for _ in range(12)]
        row = {
            'lag_1': monthly[-1], 'lag_2': monthly[-2],
            'lag_4': monthly[-4], 'lag_8': monthly[-8] if len(monthly) >= 8 else monthly[0],
            'rolling_mean_4': np.mean(monthly[-4:]),
            'rolling_std_4': np.std(monthly[-4:]),
            'trend': monthly[-1] - monthly[-4],
            'month': rng.randint(1, 13), 'week_of_year': rng.randint(1, 53),
            'quarter': rng.randint(1, 5), 'season': rng.randint(1, 5),
            'is_holiday_season': rng.randint(0, 2),
            'price_scaled': rng.uniform(0.1, 5.0),
            'product_avg_sales': np.mean(monthly),
            'product_hash': rng.uniform(0, 1),
            'category_encoded': rng.randint(0, 10),
            'product_id_encoded': rng.randint(0, 1000),
            'target': max(1, monthly[-1] * rng.uniform(0.85, 1.15)),
        }
        train_rows.append(row)

import pandas as pd
df = pd.DataFrame(train_rows)
feature_cols = [c for c in df.columns if c != 'target']
X_fc = df[feature_cols].values
y_fc = df['target'].values

# Remove outliers
q_hi = np.percentile(y_fc, 99)
mask = y_fc <= q_hi
X_fc, y_fc = X_fc[mask], y_fc[mask]

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_percentage_error

# Train RF
rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_fc, y_fc)

# Evaluate
from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(rf, X_fc, y_fc, cv=3, scoring='r2')
print(f"  RF R² CV: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

rf_payload = {
    'model': rf,
    'feature_cols': feature_cols,
    'cat_le': cat_le,
    'pid_le': pid_le,
}
with open(MODELS_DIR / "rf_forecast_model.pkl", "wb") as f:
    pickle.dump(rf_payload, f)
print("  ✅ Saved rf_forecast_model.pkl")

# ── 5. Test harmful detection with real examples ───────────────────────────────
print("\n[5/5] Testing harmful ingredient detection on sample products...")
from ml_service import svc

# Force reload the newly built models
import importlib
import ml_service as ml_mod
ml_mod.svc._load_all()

test_products = [
    {
        'name': 'Classic Moisturizer with Parabens',
        'ingredients': 'Water, Glycerin, Methylparaben, Propylparaben, Sodium Lauryl Sulfate, Fragrance'
    },
    {
        'name': 'Natural Aloe Face Cream',
        'ingredients': 'Aloe Vera, Shea Butter, Jojoba Oil, Vitamin E, Rose Water, Chamomile Extract'
    },
    {
        'name': 'Anti-Aging Retinol Night Cream',
        'ingredients': 'Retinol, Formaldehyde, Mineral Oil, Petrolatum, Lead Acetate, Mercury Compound'
    },
    {
        'name': 'Gentle Baby Wash',
        'ingredients': 'Water, Cocamidopropyl Betaine, Glycerin, Sodium Benzoate, Citric Acid'
    },
]

for p in test_products:
    result = svc.detect_harmful(ingredients=p['ingredients'], product_name=p['name'])
    print(f"\n  Product: {p['name']}")
    print(f"    Score: {result['safety_score']}/100 | Status: {result['status']}")
    print(f"    Harmful found: {result['harmful_count']} ingredients")
    for h in result['harmful_ingredients'][:3]:
        print(f"      ⚠️  {h['name']} (severity: {h['severity']}/10)")

print("\n\n=== Rebuild Complete ===")
print("Models saved to:", MODELS_DIR)
print("\nTo apply: restart the Flask backend")
print("  pkill -f 'python3 app.py' && python3 backend/app.py")
