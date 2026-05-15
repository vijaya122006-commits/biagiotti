"""
utils/review_nlp.py
=====================
Backend inference utilities for Sentiment Analysis and Skin-Type Classification.
Loads pre-trained models from backend/models/ and exposes clean Python
functions for Flask/FastAPI route handlers.

Artefacts required (in backend/models/):
  - sentiment_model.pkl
  - skin_model.pkl
"""

import pickle
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

_sentiment_payload = None
_skin_model        = None


# ─────────────────────────────────────────────────────────────────────────────
# LAZY LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def _load_sentiment() -> bool:
    global _sentiment_payload
    if _sentiment_payload is not None:
        return True
    try:
        _sentiment_payload = pickle.load(open(_MODELS_DIR / "sentiment_model.pkl", "rb"))
        logger.info("sentiment_model loaded ✔")
        return True
    except FileNotFoundError:
        logger.warning("sentiment_model.pkl not found — run train.py first.")
        return False
    except Exception as exc:
        logger.error(f"Failed to load sentiment model: {exc}")
        return False


def _load_skin() -> bool:
    global _skin_model
    if _skin_model is not None:
        return True
    try:
        _skin_model = pickle.load(open(_MODELS_DIR / "skin_model.pkl", "rb"))
        logger.info("skin_model loaded ✔")
        return True
    except FileNotFoundError:
        logger.warning("skin_model.pkl not found — run train.py first.")
        return False
    except Exception as exc:
        logger.error(f"Failed to load skin model: {exc}")
        return False


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _vader_label(compound: float) -> str:
    """Convert VADER compound score to sentiment label."""
    if compound >= 0.05:
        return "positive"
    elif compound <= -0.05:
        return "negative"
    return "neutral"


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def predict_sentiment(review_text: str) -> dict:
    """
    Predict the sentiment of a review string.

    Dispatches to the appropriate backend:
    - 'classifier' mode → TF-IDF + LogisticRegression
    - 'vader' mode       → VADER SentimentIntensityAnalyzer
    - 'textblob' mode    → TextBlob polarity

    Parameters
    ----------
    review_text : str
        Raw review text to analyse.

    Returns
    -------
    dict with keys:
        - sentiment  : "positive" | "neutral" | "negative"
        - mode       : backend used
        - compound   : float (VADER only)
        - confidence : float (classifier only)
        - scores     : dict (VADER full scores)

    Examples
    --------
    >>> from utils.review_nlp import predict_sentiment
    >>> predict_sentiment("This moisturiser is absolutely amazing!")
    {'sentiment': 'positive', 'compound': 0.785, 'mode': 'vader'}
    """
    if not review_text or not str(review_text).strip():
        return {"sentiment": "neutral", "mode": "empty"}

    if not _load_sentiment():
        # Hard fallback — inline VADER
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            ana    = SentimentIntensityAnalyzer()
            scores = ana.polarity_scores(str(review_text))
            return {
                "sentiment": _vader_label(scores["compound"]),
                "compound":  round(scores["compound"], 4),
                "scores":    {k: round(v, 4) for k, v in scores.items()},
                "mode":      "vader_inline",
            }
        except ImportError:
            return {"sentiment": "positive", "mode": "fallback_demo"}

    mode = _sentiment_payload.get("mode", "vader")

    if mode == "classifier":
        pipeline = _sentiment_payload["pipeline"]
        label    = pipeline.predict([review_text])[0]
        proba    = pipeline.predict_proba([review_text])[0]
        return {
            "sentiment":  label,
            "mode":       "classifier",
            "confidence": round(float(max(proba)), 4),
        }

    elif mode == "vader":
        analyzer = _sentiment_payload["analyzer"]
        scores   = analyzer.polarity_scores(str(review_text))
        return {
            "sentiment": _vader_label(scores["compound"]),
            "compound":  round(scores["compound"], 4),
            "scores":    {k: round(v, 4) for k, v in scores.items()},
            "mode":      "vader",
        }

    else:  # textblob
        try:
            from textblob import TextBlob
            pol = TextBlob(str(review_text)).sentiment.polarity
            return {
                "sentiment": "positive" if pol > 0.05 else ("negative" if pol < -0.05 else "neutral"),
                "polarity":  round(pol, 4),
                "mode":      "textblob",
            }
        except ImportError:
            return {"sentiment": "neutral", "mode": "none"}


def predict_skin_type(review_text: str) -> dict:
    """
    Predict the skin-type suitability label for a product based on review text.

    Parameters
    ----------
    review_text : str
        Review text or combined product description.

    Returns
    -------
    dict with keys:
        - skin_type      : str (predicted class label)
        - probabilities  : dict {class: probability}
        - confidence     : float

    Examples
    --------
    >>> from utils.review_nlp import predict_skin_type
    >>> predict_skin_type("Great for my dry skin, feels so hydrating!")
    {'skin_type': 'dry', 'probabilities': {'dry': 0.82, 'oily': 0.10, ...}, 'confidence': 0.82}
    """
    if not review_text or not str(review_text).strip():
        return {"skin_type": "unknown", "probabilities": {}, "confidence": 0.0}

    if not _load_skin():
        return {
            "skin_type":     "unknown",
            "probabilities": {},
            "confidence":    0.0,
            "note":          "skin_model.pkl not found — run train.py",
        }

    try:
        pred  = _skin_model.predict([str(review_text)])[0]
        proba = _skin_model.predict_proba([str(review_text)])[0]
        classes = _skin_model.classes_
        return {
            "skin_type":    pred,
            "probabilities": {c: round(float(p), 4) for c, p in zip(classes, proba)},
            "confidence":   round(float(max(proba)), 4),
        }
    except Exception as exc:
        logger.error(f"predict_skin_type error: {exc}")
        return {"skin_type": "unknown", "probabilities": {}, "confidence": 0.0}


def analyze_sentiment(
    product_name: str = "",
    reviews: Optional[list[str]] = None,
) -> dict:
    """
    Legacy-compatible wrapper that accepts a product name and optional review
    list and returns a structured sentiment summary dict for route handlers.

    Parameters
    ----------
    product_name : str
        Product name (used for display only).
    reviews      : list[str] | None
        Optional list of review strings. If None, returns demo data.

    Returns
    -------
    dict with keys:
        - distribution         : {'positive': %, 'neutral': %, 'negative': %}
        - skin_type_suitability: list[str]
        - review_summary       : list[dict] with insight / frequency / sentiment
    """
    if reviews is None:
        # Return demo structure when no real reviews provided
        return {
            "distribution":          {"positive": 65, "neutral": 20, "negative": 15},
            "skin_type_suitability": ["Best for Dry Skin", "Best for Sensitive Skin"],
            "review_summary": [
                {"insight": "Very hydrating, lasts all day",       "frequency": 420, "sentiment": "positive"},
                {"insight": "Caused slight redness initially",     "frequency": 45,  "sentiment": "negative"},
                {"insight": "Lightweight formula, love the scent", "frequency": 210, "sentiment": "positive"},
            ],
        }

    results   = [predict_sentiment(r) for r in reviews if r]
    sentiments = [r.get("sentiment", "neutral") for r in results]

    total = max(len(sentiments), 1)
    dist  = {
        "positive": round(100 * sentiments.count("positive") / total, 1),
        "neutral":  round(100 * sentiments.count("neutral")  / total, 1),
        "negative": round(100 * sentiments.count("negative") / total, 1),
    }

    # Skin suitability from skin model on each review
    skin_preds = [predict_skin_type(r).get("skin_type", "") for r in reviews if r]
    from collections import Counter
    skin_counts = Counter(skin_preds).most_common(3)
    skin_labels = [f"Best for {s.title()} Skin" for s, _ in skin_counts if s and s != "unknown"]

    return {
        "distribution":           dist,
        "skin_type_suitability":  skin_labels or ["Best for All Skin Types"],
        "review_summary": [
            {"insight": f"{s.title()} sentiment detected", "frequency": c, "sentiment": s}
            for s, c in Counter(sentiments).items()
        ],
    }


# Keep old call signature for backwards compatibility
Optional = Optional  # re-export type alias
