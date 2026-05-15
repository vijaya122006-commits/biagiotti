"""
utils/harmful_detector.py
===========================
Backend inference utilities for the Harmful Ingredient Detector.
Loads the pre-trained rule-based engine from backend/models/harmful_detector.pkl
and exposes clean Python functions for Flask/FastAPI route handlers.

Artefacts required (in backend/models/):
  - harmful_detector.pkl
"""

import pickle
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_detector_payload = None


def _load_detector() -> bool:
    """
    Lazy-load the harmful detector engine from disk.

    Returns
    -------
    True if loaded successfully, False otherwise.
    """
    global _detector_payload
    if _detector_payload is not None:
        return True
    try:
        _detector_payload = pickle.load(open(_MODELS_DIR / "harmful_detector.pkl", "rb"))
        logger.info("harmful_detector model loaded ✔")
        return True
    except FileNotFoundError:
        logger.warning("harmful_detector.pkl not found — run train.py first.")
        return False
    except Exception as exc:
        logger.error(f"Failed to load harmful detector: {exc}")
        return False


def _get_keywords() -> dict:
    """Return the harmful keyword dictionary from the loaded payload."""
    if not _load_detector():
        return {}
    return _detector_payload.get("harmful_keywords", {})


def compute_safety_score(ingredient_text: str) -> tuple[float, list[dict]]:
    """
    Compute a safety score (0–100) for a product.

    Algorithm
    ---------
    - For each harmful keyword detected in the ingredient string, a severity
      penalty (severity × 4 points) is deducted from 100.
    - Total penalty is capped at 100 → minimum score is 0.

    Parameters
    ----------
    ingredient_text : str
        Raw ingredient list string (comma-separated or free text).

    Returns
    -------
    (safety_score, list_of_harmful_ingredients)
        safety_score : float in [0, 100].
        list_of_harmful_ingredients : list of dicts with keys
            {'keyword', 'name', 'reason', 'severity'}.
    """
    if not ingredient_text or str(ingredient_text).strip().lower() in ("", "nan"):
        return 100.0, []

    keywords = _get_keywords()
    if not keywords:
        # Inline fallback set if model not loaded
        keywords = {
            "paraben":              {"name": "Parabens",             "reason": "Endocrine disruptor",         "severity": 7},
            "sodium lauryl sulfate": {"name": "SLS",                 "reason": "Skin irritant",               "severity": 5},
            "oxybenzone":           {"name": "Oxybenzone",           "reason": "Endocrine disruptor",         "severity": 8},
            "formaldehyde":         {"name": "Formaldehyde",         "reason": "Known carcinogen (IARC G1)", "severity": 10},
            "phthalate":            {"name": "Phthalates",           "reason": "Reproductive toxicant",       "severity": 8},
            "phenoxyethanol":       {"name": "Phenoxyethanol",       "reason": "Potential skin sensitiser",   "severity": 4},
            "petroleum":            {"name": "Petrolatum/Petroleum", "reason": "PAH contamination risk",      "severity": 5},
            "peg":                  {"name": "PEGs",                 "reason": "1,4-dioxane contamination",   "severity": 5},
        }

    text   = str(ingredient_text).lower()
    found  = []
    total_severity = 0

    for keyword, info in keywords.items():
        if keyword in text:
            found.append(
                {
                    "keyword":  keyword,
                    "name":     info["name"],
                    "reason":   info["reason"],
                    "severity": info["severity"],
                }
            )
            total_severity += info["severity"]

    penalty = min(total_severity * 4, 100)
    score   = round(max(0.0, 100.0 - penalty), 1)
    return score, found


def detect_harmful_ingredients(
    product_name: str = "",
    ingredient_text: str = "",
) -> dict:
    """
    Main route-facing function. Returns a comprehensive safety analysis dict.

    Parameters
    ----------
    product_name    : str
        Product name (used for logging / display only).
    ingredient_text : str
        Raw ingredient string to analyse. If empty, returns Safe with score 100.

    Returns
    -------
    dict with keys:
        - status         : "Safe" | "Moderate" | "Unsafe"
        - safety_score   : float [0, 100]
        - toxicity_level : int [0, 10] (normalised severity / 10)
        - ingredients    : list[dict] — detected harmful ingredients
        - recommendation : str — brief guidance string

    Examples
    --------
    >>> from utils.harmful_detector import detect_harmful_ingredients
    >>> result = detect_harmful_ingredients(
    ...     ingredient_text="Water, Methylparaben, Glycerin, Fragrance"
    ... )
    >>> print(result['status'], result['safety_score'])
    Moderate 72.0
    """
    score, found = compute_safety_score(ingredient_text)

    if score >= 80:
        status  = "Safe"
        rec     = "Product appears safe based on ingredient analysis."
    elif score >= 50:
        status  = "Moderate"
        rec     = "Some potentially concerning ingredients detected. Use with caution."
    else:
        status  = "Unsafe"
        rec     = "Multiple harmful ingredients detected. We recommend choosing an alternative."

    # Normalise severity to 0-10 toxicity level
    total_sev = sum(i["severity"] for i in found)
    toxicity  = min(10, round(total_sev / max(1, len(found)), 1)) if found else 0

    return {
        "product_name":   product_name,
        "status":         status,
        "safety_score":   score,
        "toxicity_level": toxicity,
        "ingredients":    [
            {"name": i["name"], "reason": i["reason"], "severity": i["severity"]}
            for i in found
        ],
        "recommendation": rec,
    }
