"""
test_all_models.py
==================
Run from:  biagiotti/backend/
Command:   python test_all_models.py
"""
import sys, time, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from ml_service import svc

SEP  = "=" * 70
SEP2 = "-" * 70

def hdr(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

def metric(label, value, unit=""):
    print(f"  {'':2}{label:<35} {value} {unit}")

def ok(msg):   print(f"  ✅  {msg}")
def warn(msg): print(f"  ⚠️   {msg}")
def fail(msg): print(f"  ❌  {msg}")

# ── 0. Engine Health ──────────────────────────────────────────────────────────
hdr("0 / 5  ENGINE HEALTH CHECK")
h = svc.health_check()
metric("Status",           h["status"])
metric("Components Loaded",f"{len(h['models_loaded'])} / 9")
metric("Product Registry", f"{h['product_count']} products")
metric("Loaded Modules",   ", ".join(h["models_loaded"]) or "none")
print()

if h["product_count"] == 0:
    warn("No products in registry — similarity tests will be limited")

# ── 1. Skin-Type Classifier ───────────────────────────────────────────────────
hdr("1 / 5  SKIN-TYPE CLASSIFIER  (skin_model_v2 + vectorizer_v2)")

TEST_CASES = [
    ("Oily & Acne-Prone",    "oily acne pimple breakout sebum clogged pores", "oily"),
    ("Dry & Dehydrated",     "dry flaky tight dehydrated rough scaly",         "dry"),
    ("Sensitive Redness",    "sensitive redness sting irritate itch fragile",  "sensitive"),
    ("Acne Control",         "acne blemish spot blackhead whitehead",           "acne"),
    ("Normal Balanced",      "balanced normal healthy clear skin",              "normal"),
    ("Ambiguous Short Text", "skin",                                            None),
]

correct, total = 0, 0
latencies = []
confidences = []

print(f"\n  {'Input Text':<35} {'Predicted':<12} {'Conf':>6}  {'Expected':<12} {'✓?'}")
print(f"  {SEP2}")

for label, text, expected in TEST_CASES:
    t0 = time.perf_counter()
    r  = svc.predict_skin(text)
    lat = (time.perf_counter() - t0) * 1000
    latencies.append(lat)
    confidences.append(r["confidence"])

    pred = r["skin_type"]
    conf = r["confidence"]
    match = "✅" if (expected is None or pred == expected) else "❌"
    if expected and pred == expected: correct += 1
    if expected: total += 1

    print(f"  {label:<35} {pred:<12} {conf:>6.1%}  {str(expected):<12} {match}")

print(f"\n  ── Metrics ──────────────────────────────────────────────")
metric("Accuracy (labelled cases)", f"{correct}/{total}", f"= {correct/total:.1%}")
metric("Mean Latency",              f"{sum(latencies)/len(latencies):.1f} ms")
metric("Mean Confidence",           f"{sum(confidences)/len(confidences):.1%}")
metric("Min / Max Confidence",      f"{min(confidences):.1%} / {max(confidences):.1%}")
metric("Model",                     "LinearSVC + TF-IDF Vectorizer v2")
metric("Boosting",                  "Keyword co-occurrence (oily/dry/sensitive/acne)")

# ── 2. Sentiment Analysis ─────────────────────────────────────────────────────
hdr("2 / 5  SENTIMENT ANALYSIS  (VADER + rule-based hybrid)")

SENT_CASES = [
    ("Strong Positive",  "I absolutely love this product! It's amazing, my holy grail!", "positive"),
    ("Mild Positive",    "Pretty good moisturizer, nice texture.",                        "positive"),
    ("Neutral",          "The product arrived. It is okay.",                              "neutral"),
    ("Mild Negative",    "Not what I expected, rather disappointed.",                     "negative"),
    ("Strong Negative",  "Worst product ever. Terrible breakout. Horrible. Avoid it!",   "negative"),
    ("Mixed Signal",     "Love the packaging but the formula caused breakout.",           None),
]

s_correct, s_total = 0, 0
s_latencies = []
scores = []

print(f"\n  {'Label':<22} {'Predicted':<12} {'Score':>7}  {'Expected':<12} {'✓?'}")
print(f"  {SEP2}")

for label, text, expected in SENT_CASES:
    t0 = time.perf_counter()
    r  = svc.analyze_sentiment(text)
    lat = (time.perf_counter() - t0) * 1000
    s_latencies.append(lat)
    scores.append(abs(r["score"]))

    pred = r["sentiment"]
    score = r["score"]
    match = "✅" if (expected is None or pred == expected) else "❌"
    if expected and pred == expected: s_correct += 1
    if expected: s_total += 1

    print(f"  {label:<22} {pred:<12} {score:>+7.3f}  {str(expected):<12} {match}")

print(f"\n  ── Metrics ──────────────────────────────────────────────")
metric("Accuracy (labelled cases)", f"{s_correct}/{s_total}", f"= {s_correct/s_total:.1%}")
metric("Mean Latency",              f"{sum(s_latencies)/len(s_latencies):.2f} ms")
metric("Avg |Score|",               f"{sum(scores)/len(scores):.3f}")
metric("Threshold (pos/neg)",       "> +0.15  /  < -0.15")
metric("Model",                     "VADER compound score + manual keyword boost")
metric("Polarity Boost Keywords",   "+0.4 per strong positive / -0.4 per strong negative")

# ── 3. Harmful Ingredient Detector ───────────────────────────────────────────
hdr("3 / 5  HARMFUL INGREDIENT DETECTOR  (harmful_detector.pkl)")

ING_CASES = [
    ("Clean Product",
     "aqua, glycerin, niacinamide, hyaluronic acid, ceramide np, vitamin e, aloe vera",
     "Safe"),
    ("Moderate Risk (Parabens)",
     "aqua, glycerin, methylparaben, propylparaben, cetyl alcohol",
     "Moderate"),
    ("High Risk (Multiple Hazards)",
     "aqua, formaldehyde, lead acetate, hydroquinone, mercury, quaternium-15",
     "Unsafe"),
    ("Sunscreen Common",
     "aqua, zinc oxide, titanium dioxide, avobenzone, oxybenzone, glycerin",
     None),
    ("PEG Compounds",
     "aqua, peg-100 stearate, peg-40 castor oil, glycerin, carbomer",
     None),
]

print(f"\n  {'Product':<28} {'Status':<12} {'Score':>7}  {'#Harmful':>8}  {'Expected'}")
print(f"  {SEP2}")

for label, ings, expected in ING_CASES:
    r = svc.detect_harmful(ingredients=ings, product_name=label)
    status = r["status"]
    score  = r["safety_score"]
    n_harm = r["harmful_count"]
    match  = "✅" if (expected is None or status == expected) else "❌"
    print(f"  {label:<28} {status:<12} {score:>7.1f}  {n_harm:>8}  {str(expected)} {match}")

print(f"\n  ── Metrics ──────────────────────────────────────────────")
metric("Scoring Formula",     "100 - Σ(family_severity × 3.5)")
metric("Thresholds",          "≥85 = Safe  |  60-84 = Moderate  |  <60 = Unsafe")
metric("Penalty Dedup",       "Family-based (paraben_family, peg_family, formaldehyde_family)")
metric("Keyword Dictionary",  f"{len(svc.harmful_kw)} known harmful chemicals")
metric("Model File",          "harmful_detector.pkl")

# ── 4. Product Similarity (TF-IDF Cosine) ────────────────────────────────────
hdr("4 / 5  PRODUCT SIMILARITY  (TF-IDF + Cosine Similarity Matrix)")

n_products = len(svc.product_ids)
metric("Product Registry Size",  n_products)
metric("Model",                  "TF-IDF Vectorizer + Precomputed Cosine Matrix")
metric("Category Boost",         "+20% similarity for same-category matches")

if n_products > 0:
    # Test with first, middle, last product
    test_indices = [0, n_products // 2, n_products - 1]
    print()
    for idx in test_indices:
        pid   = svc.product_ids[idx]
        pname = svc.product_names[idx]
        t0    = time.perf_counter()
        r     = svc.get_similar_products(pid=pid, top_n=3)
        lat   = (time.perf_counter() - t0) * 1000
        results = r.get("results", [])
        print(f"  Query: [{idx}] {pname[:45]}")
        if results:
            for res in results:
                print(f"    → #{res['rank']} {res['product_name'][:40]:<42} sim={res['similarity']:.4f}")
        else:
            warn("No results returned")
        print(f"    Latency: {lat:.1f}ms")
        print()

    print(f"  ── Metrics ──────────────────────────────────────────────")
    metric("Similarity Range",   "0.0 (no match) → 1.0 (identical)")
    metric("Matrix Shape",       f"{n_products} × {n_products}")
    metric("Lookup Strategy",    "PID → idx → row slice → argsort descending")
else:
    warn("Product registry empty — run sync first to populate similarity matrix")

# ── 5. Sales Forecasting (Random Forest) ─────────────────────────────────────
hdr("5 / 5  SALES FORECASTING  (Random Forest Recursive)")

FORECAST_CASES = [
    {
        "product_id": "SERUM_001", "product_name": "Vitamin C Brightening Serum",
        "price": 1499.0, "units_sold": 80, "current_stock": 200,
        "cost_price": 400, "lead_time_days": 14,
        "recent_sales": [72, 76, 80, 74, 82, 78, 85, 88, 84, 90, 87, 92]
    },
    {
        "product_id": "CLNS_002", "product_name": "Gentle Foaming Face Cleanser",
        "price": 399.0, "units_sold": 150, "current_stock": 50,
        "cost_price": 100, "lead_time_days": 7,
        "recent_sales": [140, 145, 155, 148, 152, 160, 158, 162, 155, 170, 165, 175]
    },
    {
        "product_id": "SPF_003", "product_name": "SPF 50+ Mineral Sunscreen",
        "price": 799.0, "units_sold": 60, "current_stock": 800,
        "cost_price": 200, "lead_time_days": 21,
        "recent_sales": [55, 52, 48, 44, 40, 38, 35, 33, 30, 28, 25, 22]
    },
]

print(f"\n  {'Product':<35} {'Trend':<12} {'Stockout':>9}  {'Days Inv':>9}  {'Decision'}")
print(f"  {SEP2}")

f_latencies = []
priority_scores = []
confidence_scores = []

for feat in FORECAST_CASES:
    t0 = time.perf_counter()
    r  = svc.forecast_sales(features=feat, steps=8)
    lat = (time.perf_counter() - t0) * 1000
    f_latencies.append(lat)

    if r.get("status") == "error":
        fail(f"{feat['product_name']}: {r.get('message')}")
        continue

    pname     = feat["product_name"][:33]
    trend     = r.get("trend", "?")
    stockout  = r.get("stockout_risk", "?")
    days_inv  = r.get("days_of_inventory", 0)
    decision  = r.get("decision", "?").split("|")[0].strip()[:25]
    prio      = r.get("priority_score", 0)
    conf      = r.get("confidence_score", 0)
    priority_scores.append(prio)
    confidence_scores.append(conf)

    print(f"  {pname:<35} {trend:<12} {stockout:>9}  {days_inv:>9.1f}  {decision}")
    forecast = r.get("forecast", [])
    print(f"    Forecast (8 wk):  {[round(x) for x in forecast]}")
    print(f"    Priority Score:   {prio:.1f}/100  |  Confidence: {conf:.1f}%  |  Latency: {lat:.1f}ms")
    print(f"    Reason: {r.get('reason','')[:80]}")
    print()

print(f"  ── Metrics ──────────────────────────────────────────────")
metric("Algorithm",          "Random Forest (recursive multi-step)")
metric("Feature Columns",    str(svc.rf_payload.get('feature_cols', ['N/A']) if svc.rf_payload else ['model missing']))
metric("Lag Features",       "lag_1, lag_2, lag_3, lag_4, lag_8")
metric("Rolling Features",   "rolling_mean_4, rolling_mean_8, rolling_std_4, trend_4")
metric("Calendar Features",  "month, week_of_year, quarter, season, is_holiday_season")
metric("Price Features",     "price_scaled (÷10000), margin")
metric("Smoothing",          "0.7 × RF_pred + 0.3 × lag_1")
metric("Diversity Injection","product_hash × 5  (unique per product)")
metric("Volatility (CV)",    ">0.4=high | 0.15-0.4=medium | <0.15=low")
metric("Confidence Formula", "95 - (CV × 25)  → clamped [10, 99]")
metric("Mean Latency",       f"{sum(f_latencies)/len(f_latencies):.1f} ms")
if confidence_scores:
    metric("Avg Confidence",     f"{sum(confidence_scores)/len(confidence_scores):.1f}%")
if priority_scores:
    metric("Avg Priority Score", f"{sum(priority_scores)/len(priority_scores):.1f}/100")

# ── Summary ───────────────────────────────────────────────────────────────────
hdr("SUMMARY — ALL PIPELINES")
print(f"""
  Pipeline                  Model File                     Status
  {SEP2}
  1. Skin Classifier        skin_model_v2.pkl              {'✅ LOADED' if 'skin_v2' in svc.ready else '❌ MISSING'}
     Vectorizer             vectorizer_v2.pkl              {'✅ LOADED' if 'skin_v2' in svc.ready else '❌ MISSING'}
  2. Sentiment Analysis     sentiment_model.pkl            {'✅ LOADED' if 'sentiment' in svc.ready else '❌ MISSING'}
  3. Harmful Detector       harmful_detector.pkl           {'✅ LOADED' if 'harmful' in svc.ready else '❌ MISSING'}
  4. Similarity Search      tfidf_vectorizer.pkl           {'✅ LOADED' if 'tfidf' in svc.ready else '❌ MISSING'}
                            cosine_similarity_matrix.pkl   {'✅ LOADED' if 'sim_matrix' in svc.ready else '❌ MISSING'}
                            product_id_index.pkl           {'✅ LOADED' if 'id_index' in svc.ready else '❌ MISSING'}
  5. Sales Forecasting      rf_forecast_model.pkl          {'✅ LOADED' if 'rf_forecast' in svc.ready else '❌ MISSING'}
""")
print(SEP)
print(f"  Total Components Ready: {len(svc.ready)} / 9")
print(SEP)
