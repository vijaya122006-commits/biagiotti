"""
utils/preprocess.py
=====================
Shared text and data preprocessing utilities used across the training
pipeline and inference layer.
"""

import re
import string
import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# TEXT CLEANING
# ─────────────────────────────────────────────────────────────────────────────

_PUNCTUATION_TABLE = str.maketrans("", "", string.punctuation)


def clean_text(
    text: str,
    lowercase: bool = True,
    remove_punctuation: bool = True,
    remove_extra_spaces: bool = True,
    remove_numbers: bool = False,
) -> str:
    """
    Normalise a raw text string.

    Parameters
    ----------
    text                : Input string (may be NaN-like).
    lowercase           : Convert to lower case (default True).
    remove_punctuation  : Strip punctuation characters (default True).
    remove_extra_spaces : Collapse multiple spaces (default True).
    remove_numbers      : Remove digit characters (default False).

    Returns
    -------
    Cleaned string.
    """
    if not text or str(text).strip().lower() in ("nan", "none", ""):
        return ""

    text = str(text)
    if lowercase:
        text = text.lower()
    if remove_numbers:
        text = re.sub(r"\d+", " ", text)
    if remove_punctuation:
        text = text.translate(_PUNCTUATION_TABLE)
    if remove_extra_spaces:
        text = re.sub(r"\s+", " ", text).strip()

    return text.strip()


def clean_series(series: pd.Series, **kwargs) -> pd.Series:
    """
    Vectorised wrapper around clean_text for a pandas Series.

    Parameters
    ----------
    series : pd.Series of text values.
    kwargs : Forwarded to clean_text.

    Returns
    -------
    pd.Series of cleaned strings (NaN → empty string).
    """
    return series.fillna("").astype(str).apply(lambda x: clean_text(x, **kwargs))


# ─────────────────────────────────────────────────────────────────────────────
# INGREDIENT PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_ingredients(ingredient_text: str) -> list[str]:
    """
    Parse a raw ingredient string into a list of individual ingredient tokens.

    Handles comma-separated, semicolon-separated, and free-text formats.

    Parameters
    ----------
    ingredient_text : str
        Raw ingredients string.

    Returns
    -------
    list[str] — lower-cased, stripped ingredient names.
    """
    if not ingredient_text or str(ingredient_text).strip().lower() in ("nan", "none", ""):
        return []

    text = str(ingredient_text).lower().strip()
    # Split on commas or semicolons
    parts = re.split(r"[;,]", text)
    cleaned = []
    for p in parts:
        p = p.strip().strip('"').strip("'").strip()
        if len(p) > 1:
            cleaned.append(p)
    return cleaned


def ingredients_to_text(ingredient_text: str) -> str:
    """
    Turn a raw ingredient string into clean, tokenised text suitable for TF-IDF.

    Parameters
    ----------
    ingredient_text : str
        Raw ingredients (comma/semicolon-separated).

    Returns
    -------
    Single space-separated string of ingredient tokens.
    """
    parts = parse_ingredients(ingredient_text)
    # Join them so TF-IDF treats multi-word ingredients as phrases
    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# NUMERIC CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def safe_numeric(series: pd.Series, fill: float = 0.0) -> pd.Series:
    """
    Coerce a Series to numeric, filling NaN with a given value.

    Parameters
    ----------
    series : Input series.
    fill   : Fill value for NaN / non-parseable entries (default 0.0).

    Returns
    -------
    pd.Series of float64.
    """
    return pd.to_numeric(series, errors="coerce").fillna(fill)


# ─────────────────────────────────────────────────────────────────────────────
# DATE / TIME PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def parse_dates(series: pd.Series, errors: str = "coerce") -> pd.Series:
    """
    Coerce a Series of date strings to pandas DatetimeSeries.

    Parameters
    ----------
    series : pd.Series of date strings.
    errors : 'coerce' (default) replaces unparseable with NaT.

    Returns
    -------
    pd.Series of datetime64[ns].
    """
    return pd.to_datetime(series, errors=errors)


def add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Add calendar feature columns to a DataFrame from a datetime column.

    Adds: year, month, day_of_week, week_of_year, quarter, is_weekend.

    Parameters
    ----------
    df       : DataFrame with a datetime column.
    date_col : Name of the datetime column.

    Returns
    -------
    DataFrame with additional feature columns.
    """
    df = df.copy()
    if date_col not in df.columns:
        logger.warning(f"add_time_features: column '{date_col}' not found.")
        return df

    dt = pd.to_datetime(df[date_col], errors="coerce")
    df["year"]         = dt.dt.year
    df["month"]        = dt.dt.month
    df["day_of_week"]  = dt.dt.dayofweek
    df["week_of_year"] = dt.dt.isocalendar().week.astype("Int64")
    df["quarter"]      = dt.dt.quarter
    df["is_weekend"]   = (dt.dt.dayofweek >= 5).astype(int)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# LABEL ENCODING
# ─────────────────────────────────────────────────────────────────────────────

def encode_labels(series: pd.Series) -> tuple[pd.Series, dict]:
    """
    Encode string labels to integers.

    Parameters
    ----------
    series : pd.Series of string labels.

    Returns
    -------
    (encoded_series, label_map)
    Where label_map is {original_string: integer_code}.
    """
    unique_labels = sorted(series.dropna().unique().tolist())
    label_map     = {lbl: i for i, lbl in enumerate(unique_labels)}
    encoded       = series.map(label_map)
    return encoded, label_map


# ─────────────────────────────────────────────────────────────────────────────
# SKIN-TYPE PREPROCESSING
# ─────────────────────────────────────────────────────────────────────────────

SKIN_TYPE_ALIASES: dict[str, str] = {
    # Aliases → canonical label
    "dry":           "dry",
    "dry skin":      "dry",
    "oily":          "oily",
    "oily skin":     "oily",
    "combination":   "combination",
    "combo":         "combination",
    "sensitive":     "sensitive",
    "normal":        "normal",
    "all":           "all",
    "all skin":      "all",
    "all skin types": "all",
    "unknown":       "unknown",
}


def normalise_skin_type(label: str) -> str:
    """
    Normalise a raw skin_suitability string to a canonical skin type label.

    Parameters
    ----------
    label : Raw skin type string.

    Returns
    -------
    Normalised lowercase label.
    """
    if not label or str(label).strip().lower() in ("nan", "none", ""):
        return "unknown"
    cleaned = str(label).strip().lower()
    return SKIN_TYPE_ALIASES.get(cleaned, cleaned)


def preprocess_reviews_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and preprocess the master_reviews DataFrame for model training.

    Steps:
    1. Drop rows with no review_text.
    2. Clean review_text.
    3. Normalise rating_value to [1, 5].
    4. Infer sentiment from rating if no explicit sentiment column.

    Parameters
    ----------
    df : master_reviews DataFrame.

    Returns
    -------
    Preprocessed DataFrame.
    """
    df = df.copy()

    if "review_text" in df.columns:
        df["review_text"] = clean_series(df["review_text"])
        df = df[df["review_text"].str.len() > 5]

    if "rating_value" in df.columns:
        df["rating_value"] = safe_numeric(df["rating_value"])
        df["rating_value"] = df["rating_value"].clip(1, 5)

    if "aggregate_rating" in df.columns:
        df["aggregate_rating"] = safe_numeric(df["aggregate_rating"])

    # Infer sentiment label from rating if not present
    if "sentiment" not in df.columns and "rating_value" in df.columns:
        df["sentiment"] = df["rating_value"].apply(
            lambda r: "positive" if r >= 4 else ("negative" if r <= 2 else "neutral")
        )

    return df.reset_index(drop=True)


def preprocess_master_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the master_cleaned DataFrame.

    Parameters
    ----------
    df : master_cleaned DataFrame.

    Returns
    -------
    Cleaned DataFrame.
    """
    df = df.copy()

    for col in ["product_name", "brand"]:
        if col in df.columns:
            df[col] = clean_series(df[col], remove_punctuation=False)

    if "skin_suitability" in df.columns:
        df["skin_suitability"] = df["skin_suitability"].apply(
            lambda x: normalise_skin_type(str(x))
        )

    if "total_sales" in df.columns:
        df["total_sales"] = safe_numeric(df["total_sales"])

    if "average_rating" in df.columns:
        df["average_rating"] = safe_numeric(df["average_rating"])
        df["average_rating"] = df["average_rating"].clip(0, 5)

    if "safety_score" in df.columns:
        df["safety_score"] = safe_numeric(df["safety_score"], fill=50.0)
        df["safety_score"] = df["safety_score"].clip(0, 100)

    return df.reset_index(drop=True)


def preprocess_sales_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the master_sales_cleaned DataFrame.

    Parameters
    ----------
    df : master_sales DataFrame.

    Returns
    -------
    Cleaned DataFrame.
    """
    df = df.copy()

    for date_col in ["start_date", "end_date", "order_date", "ship_date"]:
        if date_col in df.columns:
            df[date_col] = parse_dates(df[date_col])

    for qty_col in ["units_sold", "sales", "avg_daily_footfall", "sell_through_pct"]:
        if qty_col in df.columns:
            df[qty_col] = safe_numeric(df[qty_col])

    if "units_sold" in df.columns:
        df = df[df["units_sold"] >= 0]

    return df.reset_index(drop=True)
