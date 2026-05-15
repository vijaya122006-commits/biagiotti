# 🌸 Biagiotti Cosmetic Intelligence

An AI-powered retail intelligence platform for Indian cosmetic dealers — providing demand forecasting, ingredient safety analysis, sentiment insights, and inventory management.

## 🚀 Live Demo

- **Frontend:** [https://biagiotti-ui.onrender.com](https://biagiotti-ui.onrender.com)
- **Backend API:** [https://biagiotti-intelligence-jto5.onrender.com](https://biagiotti-intelligence-jto5.onrender.com)

### Demo Credentials
| Email | Password | Store |
|-------|----------|-------|
| `demo@cosmetic.ai` | `demo1234` | Biagiotti Demo Store |
| `test@biagiotti.com` | `test1234` | Biagiotti Test Shop |

---

## 🧠 Features

- **AI Demand Forecasting** — Random Forest model predicting next 1–6 months of sales
- **Ingredient Safety Analysis** — Detects harmful chemicals (Hydroquinone, Mercury, Parabens, Formaldehyde)
- **Skin Type Classification** — SVM-based classifier for oily, dry, combination, sensitive skin
- **Sentiment Analysis** — Customer review scoring with positive/neutral/negative breakdown
- **Product Similarity Engine** — Find alternative products using cosine similarity
- **Live Dashboard** — KPI cards for understock, overstock, and safety-flagged products

---

## 🗂️ Project Structure

```
biagiotti/
├── frontend/           # HTML/CSS/JS static site
│   ├── index.html      # Login page
│   ├── dashboard.html  # Main dealer dashboard
│   ├── assets/
│   │   ├── api.js      # Centralized API calls
│   │   └── app.js      # App logic
│   └── static/categories/  # Product category images
├── backend/            # Flask REST API
│   ├── app.py          # Main Flask app
│   ├── config.py       # Configuration
│   ├── create_demo_db.py  # Demo database generator (80 products)
│   ├── routes/         # API route blueprints
│   ├── services/       # ML pipeline services
│   ├── database/       # SQLAlchemy models
│   └── ml_service.py   # ML model singleton
└── render.yaml         # Render deployment config
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3, Vanilla JS, Chart.js |
| Backend | Python 3.11, Flask, SQLAlchemy |
| Database | SQLite (demo), PostgreSQL-ready |
| ML Models | scikit-learn (RF, SVM, TF-IDF) |
| Deployment | Render (Web Service + Static Site) |

---

## ⚙️ Local Development

```bash
# Backend
cd backend
pip install -r requirements.txt
python create_demo_db.py
python app.py

# Frontend
# Open frontend/index.html in browser or serve with Live Server
```

---

## 🌐 Deployment (Render)

### Backend — Web Service
- **Build:** `pip install -r backend/requirements.txt && cd backend && python create_demo_db.py`
- **Start:** `cd backend && gunicorn app:app --bind 0.0.0.0:$PORT --timeout 120`

### Frontend — Static Site
- **Publish Directory:** `frontend`

### Environment Variables
| Key | Value |
|-----|-------|
| `FLASK_ENV` | `production` |
| `SECRET_KEY` | your-secret-key |
| `API_BASE_URL` | your-render-backend-url |

---

## 📊 Demo Data

The demo database includes **80 Indian cosmetic products** across 12 categories:
- ✅ Normal stock products (routine replenishment)
- 🔴 Understock products (need immediate reorder)
- 🟠 Overstock luxury products (clearance candidates)
- ☠️ Safety-flagged products (harmful ingredients detected)

---

*Built for Indian cosmetic retail dealers — Powered by Biagiotti Intelligence Engine v3.0*