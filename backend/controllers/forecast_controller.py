# controllers/forecast_controller.py
# ─────────────────────────────────────────────────────────────────────────────
# Handles demand forecasting requests.
# Delegates to ml_service.forecast_sales() which uses the trained
# rf_forecast_model.pkl Random Forest model.
# ─────────────────────────────────────────────────────────────────────────────
import logging
from utils.response_builder import build_success_response, build_error_response
from ml_service import svc

logger = logging.getLogger("forecast_controller")


def create_forecast(request_data: dict):
    """
    POST /api/forecast
    Generate a demand forecast for a product.

    Expected request body (any of the following forms):

    Form A — list of recent weekly sales:
        { "features": [120, 135, 128, 145, 158, 163, 172, 180],
          "steps": 7,
          "product_id": "PRD_00486" }

    Form B — dict with recent_sales key:
        { "features": {
              "recent_sales":   [120, 135, ...],
              "steps":          7,
              "product_id":     "PRD_00486",
              "reference_date": "2025-09-01"
          } }

    Form C — product_id only (uses synthetic history):
        { "product_id": "PRD_00486" }
    """
    try:
        features         = request_data.get("features")
        steps            = int(request_data.get("steps", 7))
        product_id       = request_data.get("product_id") or None
        forecast_horizon = request_data.get("forecast_horizon")

        # Clamp steps to sensible range
        steps = max(1, min(steps, 52))
        
        if forecast_horizon:
            try:
                forecast_horizon = int(forecast_horizon)
            except:
                forecast_horizon = None

        # Validate list form
        if isinstance(features, list):
            try:
                features = [float(v) for v in features]
            except (TypeError, ValueError) as e:
                return build_error_response(
                    f"'features' list must contain numeric values: {e}",
                    status_code=400,
                )
            if any(v < 0 for v in features):
                return build_error_response(
                    "'features' values must be non-negative.",
                    status_code=400,
                )

        result = svc.forecast_sales(
            features         = features,
            steps            = steps,
            product_id       = str(product_id) if product_id else None,
            forecast_horizon = forecast_horizon
        )

        logger.info("Forecast → product=%s  steps=%d  fc[0]=%.2f",
                    result.get("product_id"),
                    result.get("steps", steps),
                    result["forecast"][0] if result.get("forecast") else 0)

        return build_success_response(data=result)

    except Exception as exc:
        logger.exception("create_forecast error")
        return build_error_response(str(exc), status_code=500)
