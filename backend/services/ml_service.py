# services/ml_service.py
# ─────────────────────────────────────────────────────────────────────────────
# Thin re-export shim so any module can do:
#
#   from services.ml_service import predict_skin, analyze_sentiment, ...
#
# All actual logic lives in backend/ml_service.py (one directory up).
# The singleton ``svc`` is also re-exported for callers that need it.
# ─────────────────────────────────────────────────────────────────────────────
import sys
from pathlib import Path

# Ensure the backend/ directory is on sys.path when imported from a subpackage
_BACKEND = Path(__file__).resolve().parent.parent   # → backend/
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from ml_service import (       # noqa: E402  (import after sys.path fix)
    svc,
    predict_skin,
    analyze_sentiment,
    detect_harmful,
    get_similar_products,
    forecast_sales,
)

__all__ = [
    "svc",
    "predict_skin",
    "analyze_sentiment",
    "detect_harmful",
    "get_similar_products",
    "forecast_sales",
]
