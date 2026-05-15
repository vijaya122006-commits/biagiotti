# controllers/similarity_controller.py
# ─────────────────────────────────────────────────────────────────────────────
# Handles ingredient-similarity lookups and safe-alternative recommendations.
# Both functions now delegate directly to the ml_service singleton, which uses
# the pre-trained TF-IDF vectorizer + cosine similarity matrix.
# ─────────────────────────────────────────────────────────────────────────────
import logging
from utils.response_builder import build_success_response, build_error_response
from ml_service import svc

logger = logging.getLogger("similarity_controller")


def get_similar_products(product_name: str):
    """
    GET /api/similarity/<product_name>
    Returns the top-5 most ingredient-similar products for a given product name.
    """
    try:
        result = svc.get_similar_products(
            product_index=None,
            top_n=5,
            product_name=product_name,
        )

        if "error" in result and not result.get("results"):
            logger.warning("Similarity lookup failed for '%s': %s",
                           product_name, result["error"])
            return build_error_response(
                message=f"Product '{product_name}' not found in similarity index.",
                status_code=404,
            )

        logger.info("Similarity → query='%s'  results=%d",
                    product_name, len(result["results"]))

        return build_success_response(data={
            "product":      result["query"]["product_name"] or product_name,
            "product_id":   result["query"]["product_id"],
            "alternatives": result["results"],
            "model_used":   result["model_used"],
        })

    except Exception as exc:
        logger.exception("get_similar_products error for '%s'", product_name)
        return build_error_response(str(exc), status_code=500)


def get_safe_alternatives(product_name: str):
    """
    GET /api/similarity/alternatives/<product_name>
    Returns the top-10 similar products, each enriched with a real safety score
    so the frontend can filter/sort by safety automatically.
    """
    try:
        sim_result = svc.get_similar_products(
            product_index=None,
            top_n=10,
            product_name=product_name,
        )

        safe_alternatives = []
        for alt in sim_result.get("results", []):
            # Run harmful-ingredient detection for each candidate.
            # We pass the product_name as ingredient_text as a lightweight proxy;
            # a richer implementation would look up the actual ingredient column.
            harm = svc.detect_harmful(
                product_id   = alt.get("product_id", ""),
                product_name = alt.get("product_name", ""),
            )
            alt_with_safety = {
                **alt,
                "safety_score":    harm["safety_score"],
                "safety_status":   harm["status"],
                "harmful_count":   harm["harmful_count"],
            }
            safe_alternatives.append(alt_with_safety)

        # Sort: safest first (highest safety_score), then by similarity
        safe_alternatives.sort(
            key=lambda x: (-x["safety_score"], -x["similarity"])
        )

        logger.info("Safe-alternatives → query='%s'  candidates=%d",
                    product_name, len(safe_alternatives))

        return build_success_response(data={
            "product":          sim_result["query"]["product_name"] or product_name,
            "safe_alternatives": safe_alternatives,
        })

    except Exception as exc:
        logger.exception("get_safe_alternatives error for '%s'", product_name)
        return build_error_response(str(exc), status_code=500)
