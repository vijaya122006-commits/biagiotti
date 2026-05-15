# controllers/safety_controller.py
# ─────────────────────────────────────────────────────────────────────────────
# Handles ingredient safety analysis requests.
# Delegates to ml_service.detect_harmful() which uses the trained
# harmful_detector.pkl keyword engine.
# ─────────────────────────────────────────────────────────────────────────────
import logging
from utils.response_builder import build_success_response, build_error_response
from ml_service import svc

logger = logging.getLogger("safety_controller")


def check_safety(product_name: str):
    """
    GET /api/safety/<product_name>
    Perform harmful-ingredient analysis for a product identified by name.

    Note: When ingredient_text is not supplied (name-only lookup), the engine
    still returns a score driven by keyword matching on the product name itself,
    which is useful for quick dashboard flags.  For full accuracy, pass the
    raw ingredient list via POST /harmful instead.
    """
    try:
        result = svc.detect_harmful(
            product_name    = product_name,
            ingredient_text = "",          # name-only lookup
        )

        logger.info("Safety check → product='%s'  status=%s  score=%.1f",
                    product_name, result["status"], result["safety_score"])

        return build_success_response(data={
            "product":          product_name,
            "safety_analysis":  result,
        })

    except Exception as exc:
        logger.exception("check_safety error for '%s'", product_name)
        return build_error_response(str(exc), status_code=500)
