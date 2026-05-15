"""
utils/forecasting.py
======================
Backend inference utilities for Demand Forecasting.
Loads pre-trained ARIMA and RandomForest models from backend/models/ and
exposes functions usable in Flask/FastAPI route handlers.

Artefacts required (in backend/models/):
  - arima_model.pkl
  - rf_forecast_model.pkl
"""

import pickle
import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_arima_payload = None
_rf_payload    = None


# ─────────────────────────────────────────────────────────────────────────────
# LAZY LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_arima() -> bool:
    global _arima_payload
    if _arima_payload is not None:
        return True
    try:
        _arima_payload = pickle.load(open(_MODELS_DIR / "arima_model.pkl", "rb"))
        logger.info("arima_model loaded ✔")
        return True
    except FileNotFoundError:
        logger.warning("arima_model.pkl not found — run train.py first.")
        return False
    except Exception as exc:
        logger.error(f"Failed to load ARIMA model: {exc}")
        return False


def _load_rf() -> bool:
    global _rf_payload
    if _rf_payload is not None:
        return True
    try:
        _rf_payload = pickle.load(open(_MODELS_DIR / "rf_forecast_model.pkl", "rb"))
        logger.info("rf_forecast_model loaded ✔")
        return True
    except FileNotFoundError:
        logger.warning("rf_forecast_model.pkl not found — run train.py first.")
        return False
    except Exception as exc:
        logger.error(f"Failed to load RF forecast model: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _make_seasonal_pattern(forecast: list[float]) -> str:
    """Generate a human-readable seasonal pattern sentence from a forecast list."""
    if not forecast:
        return "Insufficient data for pattern analysis."
    avg   = np.mean(forecast)
    trend = forecast[-1] - forecast[0] if len(forecast) > 1 else 0

    if trend > avg * 0.1:
        direction = "upward"
        action    = "Consider increasing inventory."
    elif trend < -avg * 0.1:
        direction = "downward"
        action    = "Monitor closely; consider promotional activity."
    else:
        direction = "stable"
        action    = "Maintain current stock levels."

    return f"Forecast shows a {direction} trend over the next {len(forecast)} weeks. {action}"


def _recommendation(forecast: list[float], current_stock: Optional[int] = None) -> str:
    """Generate a restocking recommendation from a 7-week forecast."""
    if not forecast:
        return "Insufficient data."
    peak    = max(forecast)
    avg     = np.mean(forecast)
    total   = sum(forecast)

    if current_stock is not None:
        weeks_cover = current_stock / avg if avg > 0 else 0
        if weeks_cover < 2:
            return "Buy More — stock critically low relative to forecasted demand."
        elif weeks_cover > 6:
            return "Hold — current stock exceeds 6-week forecast demand."
        else:
            return "Monitor — stock levels adequate for near-term demand."
    else:
        if avg > 500:
            return "Buy More — high forecasted demand."
        elif avg > 100:
            return "Maintain stock."
        else:
            return "Low demand forecast — review slow-moving inventory."


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def forecast_arima(product_id: str, steps: int = 7) -> dict:
    """
    Generate a multi-step ARIMA demand forecast for a product.

    Parameters
    ----------
    product_id : str
        Product ID to forecast (must exist in trained ARIMA model set).
    steps      : int
        Number of weeks ahead to forecast (default 7).

    Returns
    -------
    dict:
        - product_id        : str
        - forecast          : list[float] — weekly unit demand predictions
        - model_available   : bool
        - seasonal_pattern  : str
        - recommendation    : str

    Examples
    --------
    >>> from utils.forecasting import forecast_arima
    >>> forecast_arima('PRD_00179', steps=4)
    {'product_id': 'PRD_00179', 'forecast': [3210.0, 3350.1, ...], ...}
    """
    if not _load_arima():
        forecast = [float(np.random.randint(50, 200)) for _ in range(steps)]
        return {
            "product_id":      product_id,
            "forecast":        forecast,
            "model_available": False,
            "seasonal_pattern": _make_seasonal_pattern(forecast),
            "recommendation":  _recommendation(forecast),
            "note":            "Demo data — run train.py to generate real model.",
        }

    models = _arima_payload.get("models", {})
    if str(product_id) not in models:
        # Fallback: use aggregate / any available model
        available = list(models.keys())
        if not available:
            forecast = [float(np.random.randint(50, 200)) for _ in range(steps)]
            return {
                "product_id":      product_id,
                "forecast":        forecast,
                "model_available": False,
                "seasonal_pattern": _make_seasonal_pattern(forecast),
                "recommendation":  _recommendation(forecast),
            }
        fallback_id = available[0]
        logger.warning(f"product_id '{product_id}' not in ARIMA models — using '{fallback_id}'.")
        fitted = models[fallback_id]
    else:
        fitted = models[str(product_id)]

    try:
        forecast_values = fitted.forecast(steps=steps)
        forecast = [round(max(0.0, float(v)), 2) for v in forecast_values]
    except Exception as exc:
        logger.error(f"ARIMA forecast error: {exc}")
        forecast = [0.0] * steps

    return {
        "product_id":       product_id,
        "forecast":         forecast,
        "model_available":  True,
        "seasonal_pattern": _make_seasonal_pattern(forecast),
        "recommendation":   _recommendation(forecast),
    }


def forecast_rf(
    recent_sales: list[float],
    steps: int = 7,
    reference_date: Optional[str] = None,
) -> dict:
    """
    Generate a demand forecast using the Random Forest model.
    Requires a short history of recent weekly sales to engineer features.

    Parameters
    ----------
    recent_sales   : list[float]
        At least 8 weeks of recent sales values (most recent last).
    steps          : int
        Number of weeks ahead to forecast (default 7).
    reference_date : str | None
        ISO date string for the last known data point (defaults to today).

    Returns
    -------
    dict:
        - forecast         : list[float]
        - model_available  : bool
        - seasonal_pattern : str
        - recommendation   : str

    Examples
    --------
    >>> from utils.forecasting import forecast_rf
    >>> forecast_rf([120, 135, 140, 128, 160, 155, 170, 180], steps=4)
    {'forecast': [185.2, 190.1, ...], 'model_available': True, ...}
    """
    if not _load_rf():
        forecast = [round(float(np.mean(recent_sales or [100])) * (1 + 0.03 * i), 1)
                    for i in range(steps)]
        return {
            "forecast":        forecast,
            "model_available": False,
            "seasonal_pattern": _make_seasonal_pattern(forecast),
            "recommendation":  _recommendation(forecast),
            "note":            "Demo data — run train.py to generate real model.",
        }

    rf           = _rf_payload["model"]
    feature_cols = _rf_payload["feature_cols"]

    if len(recent_sales) < 8:
        logger.warning("Too few recent sales values for RF feature engineering — padding with mean.")
        pad = [float(np.mean(recent_sales))] * (8 - len(recent_sales))
        recent_sales = pad + list(recent_sales)

    ref_date = pd.Timestamp(reference_date) if reference_date else pd.Timestamp.today()

    # Generate future feature rows
    history = list(recent_sales)
    forecast = []

    for step in range(steps):
        future_date = ref_date + pd.DateOffset(weeks=step + 1)
        row = {
            "month":          future_date.month,
            "week_of_year":   future_date.isocalendar()[1],
            "quarter":        future_date.quarter,
            "season":         (future_date.quarter - 1) % 4 + 1,
            "is_holiday":     int(future_date.quarter == 4),
            "lag_1":          history[-1],
            "lag_2":          history[-2] if len(history) >= 2 else history[-1],
            "lag_4":          history[-4] if len(history) >= 4 else history[-1],
            "lag_8":          history[-8] if len(history) >= 8 else history[-1],
            "rolling_mean_4": float(np.mean(history[-4:])),
        }
        x = np.array([[row[c] for c in feature_cols]])
        pred = float(max(0.0, rf.predict(x)[0]))
        forecast.append(round(pred, 2))
        history.append(pred)  # use prediction as next lag

    return {
        "forecast":        forecast,
        "model_available": True,
        "seasonal_pattern": _make_seasonal_pattern(forecast),
        "recommendation":  _recommendation(forecast),
    }


def forecast_sales(
    product_id_or_csv: Union[str, list, None] = None,
    steps: int = 7,
    method: str = "auto",
) -> dict:
    """
    Unified legacy-compatible forecasting entry point.

    Dispatches to ARIMA (if product_id given) or RF (if a list of recent
    sales values given, or product not found in ARIMA).

    Parameters
    ----------
    product_id_or_csv : str | list | None
        Either a product ID string, a list of recent weekly units, or None.
    steps             : int
        Weeks to forecast.
    method            : "auto" | "arima" | "rf"
        Force a specific method or 'auto' to choose the best available.

    Returns
    -------
    dict with:
        - forecast_30_days    : list[float] (up to 4–7 weeks of forecasts)
        - seasonal_pattern    : str
        - recommendation      : str
        - graph_base64        : str  (placeholder for chart image)
        - method_used         : str
    """
    # Determine which method / input to use
    if method == "arima" or (method == "auto" and isinstance(product_id_or_csv, str)):
        pid    = product_id_or_csv
        result = forecast_arima(pid or "UNKNOWN", steps=steps)
        method_used = "arima"

    elif method == "rf" or isinstance(product_id_or_csv, list):
        sales  = product_id_or_csv if isinstance(product_id_or_csv, list) else []
        result = forecast_rf(sales or [100] * 8, steps=steps)
        method_used = "rf"

    else:
        # Pure demo fallback
        result = {
            "forecast":        [45, 48, 55, 53, 60, 65, 75][:steps],
            "model_available": False,
            "seasonal_pattern": "Heading into holiday season peak. Expect a 15% surge.",
            "recommendation":  "Buy More",
        }
        method_used = "demo"

    return {
        "forecast_30_days":  result.get("forecast", []),
        "seasonal_pattern":  result.get("seasonal_pattern", ""),
        "recommendation":    result.get("recommendation", ""),
        "graph_base64":      "",   # filled in by route handler using matplotlib
        "method_used":       method_used,
        "model_available":   result.get("model_available", False),
    }
