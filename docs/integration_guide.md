# Biagiotti — ML Integration Guide

> **System:** Cosmetic Market Intelligence Platform  
> **Stack:** HTML/JS Frontend → Flask Backend → scikit-learn / VADER / statsmodels ML Layer  
> **Version:** 1.0.0 · April 2026

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Frontend → Backend API Calls](#2-frontend--backend-api-calls)
3. [Backend → ML Model Inference](#3-backend--ml-model-inference)
4. [Running Flask & Testing APIs](#4-running-flask--testing-apis)
5. [Debugging Inference Issues](#5-debugging-inference-issues)
6. [Full End-to-End Flow Example](#6-full-end-to-end-flow-example)

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        BROWSER (Frontend)                        │
│  safety.html  skin.html  similarity.html  forecast.html         │
│                    ↕  assets/api.js  (fetch)                     │
└────────────────────────┬────────────────────────────────────────┘
                         │  HTTP POST / GET (JSON)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FLASK BACKEND  (app.py)                       │
│                                                                  │
│   POST /predict-skin       POST /sentiment                       │
│   POST /harmful            POST /similar-products                │
│   POST /forecast           GET  /api/health                      │
│                                                                  │
│            ↕  from ml_service import svc                         │
└────────────────────────┬────────────────────────────────────────┘
                         │  Python function calls (in-process)
                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ML SERVICE  (ml_service.py)                   │
│             Singleton — models loaded ONCE at startup            │
│                                                                  │
│   tfidf_vectorizer.pkl       cosine_similarity_matrix.pkl        │
│   harmful_detector.pkl       skin_model.pkl                      │
│   sentiment_model.pkl        rf_forecast_model.pkl               │
└─────────────────────────────────────────────────────────────────┘
```

**Key design principle:** Models are never reloaded per-request. The `_MLService` singleton loads all artefacts once when `app.py` starts, keeping every inference call to **< 100 ms**.

---

## 2. Frontend → Backend API Calls

### 2.1 The API Client (`assets/api.js`)

Every HTML page imports a single helper file that wraps all five ML endpoints:

```html
<!-- Add before closing </body> on any page -->
<script src="assets/api.js"></script>
```

`window.API` is then available globally:

```js
// Skin-type prediction
const result = await API.predictSkin("This serum is great for dry skin.");
console.log(result.skin_type);   // "dry"
console.log(result.confidence);  // 0.83

// Sentiment analysis
const s = await API.analyzeSentiment("I absolutely love this product!");
console.log(s.sentiment);  // "positive"
console.log(s.compound);   // 0.85

// Harmful ingredient detection
const h = await API.detectHarmful("PRD_001", "Water, Methylparaben, Oxybenzone");
console.log(h.safety_score);  // 28.0
console.log(h.status);        // "Unsafe"

// Similar products
const sim = await API.getSimilarProducts(0, 5);
sim.results.forEach(p => console.log(p.product_name, p.similarity));

// Demand forecast
const fc = await API.forecastSales([120, 135, 128, 145, 158, 163, 172, 180], 6);
console.log(fc.forecast);         // [32.4, 31.6, 26.3, ...]
console.log(fc.recommendation);   // "Buy More — forecasted demand is rising."
```

### 2.2 What `fetch()` sends

Every call is a `POST` with `Content-Type: application/json`:

```
POST http://localhost:5050/predict-skin
Content-Type: application/json

{ "text": "This serum is great for dry skin." }
```

### 2.3 Error Handling

`api.js` catches all failures and surfaces them as structured `ApiError` objects:

```js
try {
    const result = await API.predictSkin(text);
} catch (err) {
    console.error(err.message);  // Human-readable description
    console.error(err.status);   // HTTP status code (0 = network error)
    console.error(err.isNetwork); // true if server unreachable
}
```

Transient 5xx errors are **automatically retried twice** with exponential backoff before throwing.

### 2.4 Changing the Backend URL

The default URL is `http://localhost:5050`. Override it globally before any call:

```js
API.config.BASE_URL = "https://your-production-domain.com";
```

Or set it via a window variable before the script loads:

```html
<script>window.__API_BASE_URL__ = "https://your-domain.com";</script>
<script src="assets/api.js"></script>
```

---

## 3. Backend → ML Model Inference

### 3.1 File Structure

```
backend/
├── app.py                  ← Flask app, all API endpoints
├── ml_service.py           ← Singleton ML service (model loading + inference)
├── services/
│   ├── __init__.py
│   └── ml_service.py       ← Re-export shim (from ml_service import ...)
├── models/
│   ├── tfidf_vectorizer.pkl
│   ├── cosine_similarity_matrix.pkl
│   ├── product_id_index.pkl
│   ├── harmful_detector.pkl
│   ├── skin_model.pkl
│   ├── sentiment_model.pkl
│   └── rf_forecast_model.pkl
└── train.py                ← Run this to (re)train and save all models
```

### 3.2 How `app.py` Calls the ML Service

```python
# app.py — endpoint example
from services.ml_service import predict_skin

@app.route("/predict-skin", methods=["POST"])
@require_json
@timed
def endpoint_predict_skin():
    text   = request.get_json()["text"]
    result = predict_skin(text)          # ← calls ml_service singleton
    return _ok(result)
```

### 3.3 How `ml_service.py` Works

**At startup** (import time), all models are loaded from `backend/models/`:

```python
# ml_service.py — simplified
class _MLService:
    def __init__(self):
        self._load_all()          # runs ONCE when Python imports this file

    def predict_skin(self, text):
        cleaned = _clean(text)
        pred    = self.skin_model.predict([cleaned])[0]
        proba   = self.skin_model.predict_proba([cleaned])[0]
        return {"skin_type": pred, "confidence": float(max(proba)), ...}

svc = _MLService()               # ← singleton; shared across all requests
```

**Every subsequent request** is just a function call — no disk I/O, no reloading.

### 3.4 Model → Endpoint Mapping

| Endpoint | Model file | Function |
|---|---|---|
| `POST /predict-skin` | `skin_model.pkl` | `svc.predict_skin(text)` |
| `POST /sentiment` | `sentiment_model.pkl` | `svc.analyze_sentiment(text)` |
| `POST /harmful` | `harmful_detector.pkl` | `svc.detect_harmful(...)` |
| `POST /similar-products` | `tfidf_vectorizer.pkl` + `cosine_similarity_matrix.pkl` | `svc.get_similar_products(...)` |
| `POST /forecast` | `rf_forecast_model.pkl` | `svc.forecast_sales(...)` |
| `GET /api/health` | *(all)* | `svc.health_check()` |

---

## 4. Running Flask & Testing APIs

### 4.1 First-Time Setup

```bash
# 1. Install Python dependencies
pip3 install flask flask-cors scikit-learn pandas numpy \
             vaderSentiment statsmodels joblib

# 2. Train all ML models (generates backend/models/*.pkl)
cd backend
python3 train.py

# 3. Verify all models load correctly
python3 test_models.py

# 4. (Optional) Smoke-test the service layer alone
python3 ml_service.py
```

### 4.2 Start the Server

```bash
# macOS: port 5000 conflicts with AirPlay Receiver — use 5050
PORT=5050 python3 backend/app.py

# Or set a custom port
PORT=8080 python3 backend/app.py
```

Expected startup output:
```
[INFO]  ml_service  [tfidf]       loaded  214.7 KB  in  928 ms
[INFO]  ml_service  [sim_matrix]  loaded  9225.6 KB in    3 ms
[INFO]  ml_service  [skin]        loaded  694.7 KB  in  122 ms
...
[INFO]  ml_service  MLService ready — 7/7 artefacts loaded (1181 ms)
[INFO]  app         Starting Cosmetic Market Intelligence API
[INFO]  app         Port : 5050 | Debug : True | Models : 7 loaded
```

### 4.3 Test APIs with curl

```bash
BASE="http://localhost:5050"

# Health check
curl $BASE/api/health

# Skin prediction
curl -s -X POST $BASE/predict-skin \
  -H "Content-Type: application/json" \
  -d '{"text":"Perfect for dry, flaky skin"}' | python3 -m json.tool

# Sentiment
curl -s -X POST $BASE/sentiment \
  -H "Content-Type: application/json" \
  -d '{"text":"Absolutely love this serum!"}' | python3 -m json.tool

# Harmful ingredients
curl -s -X POST $BASE/harmful \
  -H "Content-Type: application/json" \
  -d '{"ingredient_text":"Water, Methylparaben, Oxybenzone","product_name":"Test Cream"}' \
  | python3 -m json.tool

# Similar products (by index)
curl -s -X POST $BASE/similar-products \
  -H "Content-Type: application/json" \
  -d '{"product_index":0,"top_n":3}' | python3 -m json.tool

# Sales forecast
curl -s -X POST $BASE/forecast \
  -H "Content-Type: application/json" \
  -d '{"features":[120,135,128,145,158,163,172,180],"steps":6}' \
  | python3 -m json.tool
```

### 4.4 Test in Browser Console

Load any page (e.g. `safety.html`) and open DevTools → Console:

```js
// Run the full built-in example suite
await window._runApiExamples();

// Or call any function directly
await API.checkHealth();
await API.predictSkin("great for sensitive skin");
```

Or append `?debug=1` to any page URL to auto-run the example block on load:
```
file:///path/to/skin.html?debug=1
```

---

## 5. Debugging Inference Issues

### 5.1 Models Not Loading

**Symptom:** `health_check()` returns `"status": "degraded"` or a model key is in `models_missing`.

```bash
# Check which .pkl files exist
ls backend/models/

# Re-run training for missing models
cd backend && python3 train.py

# Verify individually
python3 test_models.py
```

### 5.2 Wrong Predictions

**Symptom:** skin_type is always the same label, or sentiment is always neutral.

| Check | Command |
|---|---|
| Verify model's training classes | `python3 -c "import pickle; m=pickle.load(open('backend/models/skin_model.pkl','rb')); print(m.classes_)"` |
| Check review text is not empty | Ensure the `text` field is non-empty before calling API |
| Check sentiment mode | `curl .../api/ml/info` — look for `"sentiment_mode"` |
| Re-train with more data | Add more labelled reviews to `master_reviews_cleaned.csv` → re-run `train.py` |

### 5.3 CORS Errors in Browser

**Symptom:** `Access-Control-Allow-Origin` error in DevTools.

```python
# app.py — already configured, but verify:
CORS(app, resources={r"/*": {"origins": "*"}})   # dev: allow all
# Production — restrict to your domain:
# CORS(app, resources={r"/*": {"origins": "https://yourdomain.com"}})
```

### 5.4 Slow Responses

**Symptom:** API calls take > 2 seconds.

- The **first request** after startup may be slow if the OS is still paging `.pkl` files into memory — this is normal.
- All subsequent requests should be **< 100 ms** (models are in RAM).
- If still slow, check `processing_ms` in the JSON response. Values > 500 ms suggest a model size issue — consider retraining with fewer TF-IDF features.

### 5.5 Port Already in Use

```bash
# Find the process on port 5050
lsof -i :5050
kill -9 <PID>

# Or just use a different port
PORT=8080 python3 backend/app.py
```

### 5.6 Quick Health Check from Python

```python
import requests
r = requests.get("http://localhost:5050/api/health")
print(r.json())
# {"status": "ok", "models_loaded": [...], "product_count": 994}
```

---

## 6. Full End-to-End Flow Example

### Flow: User pastes ingredients → Backend scans → ML scores → Frontend renders

```
User (safety.html)
│
│  1. Types ingredient list into <textarea>
│     "Water, Methylparaben, Propylparaben, Glycerin, Oxybenzone"
│
│  2. Clicks "Analyze Safety"
│     → safetyAnalyzeBtn click listener fires
│
│  3. api.js sends HTTP request
│     POST http://localhost:5050/harmful
│     { "ingredient_text": "...", "product_name": "Rose Blush" }
│
└──────────────► Flask (app.py)
                 │
                 │  4. @require_json decorator validates Content-Type
                 │  5. endpoint_harmful() extracts ingredient_text
                 │  6. Calls:  detect_harmful(ingredient_text="...")
                 │
                 └──────────────► ml_service.py (_MLService.detect_harmful)
                                  │
                                  │  7. Iterates harmful_kw dict (26 keywords)
                                  │  8. Matches: Methylparaben ✓ (sev 7)
                                  │             Propylparaben  ✓ (sev 7)
                                  │             Oxybenzone     ✓ (sev 8)
                                  │  9. Computes:
                                  │     safety_score = 100 - min(penalty, 100) = 0
                                  │     toxicity_level = mean(7,7,8) = 7.3
                                  │     status = "Unsafe"
                                  │
                                  └──────────────► Returns dict to app.py
                 │
                 │  10. _ok(result) wraps in success envelope
                 │      { status:"success", data:{...}, processing_ms:0.3 }
                 │
                 └──────────────► JSON response → browser
│
│  11. api.js parses response.data
│  12. renderResults(data) updates DOM:
│      • safetyStatusCard border → var(--danger)
│      • safetyBadge → "Unsafe"
│      • toxicityBar width → 73%  (7.3/10 × 100)
│      • safetyScoreBar width → 0%
│      • harmfulList → 3 <li> items with names + reasons
│      • ingredientTableBody → row per ingredient, highlighted in red
│
▼
User sees live safety analysis in < 1 second ✓
```

### Sequence Diagram (text form)

```
Browser          api.js          Flask           ml_service
   │                │               │                │
   │─ click ───────►│               │                │
   │                │─ POST /harmful►│                │
   │                │               │─ detect_harmful►│
   │                │               │                │─ score ingredients
   │                │               │                │◄─ {score, found[]}
   │                │               │◄─ {data:{...}} │
   │                │◄─ JSON ───────│                │
   │◄─ renderResults│               │                │
   │                │               │                │
```

---

## Quick Reference Card

```
┌──────────────────────────────────────────────────────────┐
│  TRAIN          cd backend && python3 train.py           │
│  TEST MODELS    python3 test_models.py                   │
│  RUN SERVER     PORT=5050 python3 backend/app.py         │
│  HEALTH CHECK   curl http://localhost:5050/api/health    │
│  MODEL INFO     curl http://localhost:5050/api/ml/info   │
│  JS CONSOLE     await window._runApiExamples()           │
│  JS DEBUG PAGE  open any page with ?debug=1 in URL       │
└──────────────────────────────────────────────────────────┘

API Endpoints
  POST /predict-skin         → { skin_type, confidence, probabilities }
  POST /sentiment            → { sentiment, compound, scores, mode }
  POST /harmful              → { safety_score, status, harmful_ingredients[] }
  POST /similar-products     → { query:{}, results:[{rank,similarity,...}] }
  POST /forecast             → { forecast:[], recommendation, seasonal_pattern }
  GET  /api/health           → { status:"ok", models_loaded:[], product_count }
  GET  /api/ml/info          → { tfidf_features, sim_matrix_shape, skin_classes, ... }

Model Files  (backend/models/)
  tfidf_vectorizer.pkl          214 KB   TF-IDF ingredient vectors
  cosine_similarity_matrix.pkl  9.2 MB   994×994 sparse cosine matrix
  product_id_index.pkl           50 KB   product_id ↔ matrix row mapping
  harmful_detector.pkl            2 KB   harmful keyword definitions
  skin_model.pkl                695 KB   Logistic Regression (dry/oily/normal/sensitive)
  sentiment_model.pkl           823 KB   VADER SentimentIntensityAnalyzer
  rf_forecast_model.pkl         4.0 MB   Random Forest demand forecaster
```

---

*Generated automatically · Biagiotti Cosmetic Intelligence System · 2026*
