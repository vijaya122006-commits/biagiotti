# controllers/sentiment_controller.py
# ─────────────────────────────────────────────────────────────────────────────
# Handles skin-type NLP analysis requests.
# Delegates to ml_service for both sentiment analysis and skin-type prediction.
# ─────────────────────────────────────────────────────────────────────────────
import logging
from utils.response_builder import build_success_response, build_error_response
from ml_service import svc

logger = logging.getLogger("sentiment_controller")


def get_skin_analysis(product_name: str):
    """
    GET /api/skin-analysis/<product_name>
    Run sentiment analysis + skin-type prediction using the product name as
    the input text.  For richer results, clients should POST raw review text
    directly to /sentiment or /predict-skin.
    """
    try:
        # Use the product name as the text signal for both models
        skin_result      = svc.predict_skin(product_name)
        sentiment_result = svc.analyze_sentiment(product_name)

        logger.info("Skin analysis → product='%s'  skin=%s  sentiment=%s",
                    product_name,
                    skin_result.get("skin_type"),
                    sentiment_result.get("sentiment"))

        return build_success_response(data={
            "product":        product_name,
            "skin_type":      skin_result["skin_type"],
            "skin_confidence": skin_result["confidence"],
            "probabilities":  skin_result.get("probabilities", {}),
            "sentiment":      sentiment_result["sentiment"],
            "sentiment_mode": sentiment_result.get("mode"),
            "compound":       sentiment_result.get("compound"),
        })

    except Exception as exc:
        logger.exception("get_skin_analysis error for '%s'", product_name)
        return build_error_response(str(exc), status_code=500)
