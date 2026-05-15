"""
=============================================================================
evaluate_models.py  —  Cosmetic Market Intelligence | Full Evaluation Suite
=============================================================================
Evaluates all 6 trained models against real cleaned datasets.
Outputs:
  - Terminal metrics summary
  - backend/evaluation_outputs/*.png  (confusion matrix, forecast, etc.)

Usage:
    cd backend/
    python evaluate_models.py
=============================================================================
"""

from __future__ import annotations

import os
import sys
import pickle
import warnings
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (no display needed)
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  — all relative to this file's directory (backend/)
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).resolve().parent
MODELS_DIR    = BASE_DIR / "models"
DATA_DIR      = BASE_DIR / "data" / "cleaned"
OUTPUT_DIR    = BASE_DIR / "evaluation_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR HELPERS
# ─────────────────────────────────────────────────────────────────────────────

GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}✔{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}⚠{RESET} {msg}")
def err(msg):  print(f"  {RED}✘{RESET} {msg}")
def sep(title=""):
    print(f"\n{CYAN}{'═'*70}{RESET}")
    if title:
        print(f"{BOLD}{CYAN}  {title}{RESET}")
    print()

# ─────────────────────────────────────────────────────────────────────────────
# DATASET LOADER
# ─────────────────────────────────────────────────────────────────────────────

def load_csv(name: str, nrows=None) -> pd.DataFrame:
    path = DATA_DIR / name
    if not path.exists():
        warn(f"{name} not found at {path}")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, nrows=nrows, on_bad_lines="skip",
                         encoding="utf-8", low_memory=False)
        ok(f"Loaded {name}  → {len(df):,} rows × {df.shape[1]} cols")
        return df
    except Exception as exc:
        err(f"Could not load {name}: {exc}")
        return pd.DataFrame()


def load_pkl(name: str):
    path = MODELS_DIR / name
    if not path.exists():
        err(f"Model file not found: {path}")
        return None
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        size_kb = path.stat().st_size / 1024
        ok(f"Loaded {name}  ({size_kb:,.0f} KB)")
        return obj
    except Exception as exc:
        err(f"Could not load {name}: {exc}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# HELPER — find a column by aliases
# ─────────────────────────────────────────────────────────────────────────────

def find_col(df: pd.DataFrame, aliases: list[str]) -> str | None:
    for a in aliases:
        if a in df.columns:
            return a
    return None

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — LOAD DATASETS
# ─────────────────────────────────────────────────────────────────────────────

sep("STEP 1 — LOADING DATASETS")
df_master  = load_csv("master_cleaned.csv")
df_reviews = load_csv("master_reviews_cleaned.csv", nrows=50_000)
df_sales   = load_csv("master_sales_cleaned.csv")

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────

sep("STEP 2 — LOADING MODELS")
skin_model       = load_pkl("skin_model.pkl")
sentiment_model  = load_pkl("sentiment_model.pkl")
rf_model_payload = load_pkl("rf_forecast_model.pkl")
arima_payload    = load_pkl("arima_model.pkl")
tfidf_vec        = load_pkl("tfidf_vectorizer.pkl")
cosine_matrix    = load_pkl("cosine_similarity_matrix.pkl")
harmful_detector = load_pkl("harmful_detector.pkl")
id_index         = load_pkl("product_id_index.pkl")

# ─────────────────────────────────────────────────────────────────────────────
# RESULTS ACCUMULATOR
# ─────────────────────────────────────────────────────────────────────────────

results: dict = {}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — SKIN TYPE CLASSIFIER
# ─────────────────────────────────────────────────────────────────────────────

sep("MODEL 1 — SKIN TYPE CLASSIFIER")

try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (accuracy_score, precision_score,
                                  recall_score, f1_score,
                                  classification_report, confusion_matrix)

    # Detect text column
    text_col = find_col(df_reviews,
                        ["review_text", "review", "comment", "feedback",
                         "text", "review_body", "review_comment"])

    # Detect label column
    label_col = find_col(df_reviews,
                         ["skin_suitability", "skin_type", "skin",
                          "skin_category"])

    if skin_model is None:
        err("skin_model.pkl missing — skipping")
    elif text_col is None:
        warn("No review text column found in reviews dataset — trying master_cleaned")
        text_col  = find_col(df_master, ["review_text"])
        label_col = find_col(df_master, ["skin_suitability", "skin_type"])

    if skin_model and text_col:
        # Build data
        src_df = df_reviews if text_col in df_reviews.columns else df_master

        # If label col missing, generate proxy labels using the model itself
        if label_col and label_col in src_df.columns:
            data = src_df[[text_col, label_col]].dropna()
            data = data[data[text_col].astype(str).str.len() > 5]
            data = data.sample(min(8000, len(data)), random_state=42)
            X = data[text_col].astype(str).tolist()
            y_true = data[label_col].astype(str).tolist()
        else:
            # No label col — use master_cleaned's skin_suitability
            skill_df = df_master[["product_name_clean", "skin_suitability"]].dropna()
            skill_df = skill_df[skill_df["product_name_clean"].astype(str).str.len() > 3]
            X = skill_df["product_name_clean"].astype(str).tolist()
            y_true = skill_df["skin_suitability"].astype(str).tolist()
            warn("No review_text label col — using product_name_clean for proxy eval")

        X_train, X_test, y_train, y_test = train_test_split(
            X, y_true, test_size=0.2, random_state=42, stratify=y_true
            if len(set(y_true)) > 1 else None
        )

        print(f"  Test samples : {len(X_test):,}")
        print(f"  Classes      : {sorted(set(y_true))}")

        y_pred = skin_model.predict(X_test)

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        results["Skin Model"] = {
            "Accuracy": f"{acc*100:.2f}%",
            "Precision": f"{prec*100:.2f}%",
            "Recall": f"{rec*100:.2f}%",
            "F1-Score": f"{f1*100:.2f}%",
        }

        print(f"\n  Accuracy  : {BOLD}{acc*100:.2f}%{RESET}")
        print(f"  Precision : {prec*100:.2f}%")
        print(f"  Recall    : {rec*100:.2f}%")
        print(f"  F1-Score  : {f1*100:.2f}%")
        print(f"\n{classification_report(y_test, y_pred, zero_division=0)}")

        # ── Confusion Matrix Plot ──────────────────────────────────────────
        classes = sorted(set(list(y_test) + list(y_pred)))
        cm = confusion_matrix(y_test, y_pred, labels=classes)

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        fig.colorbar(im, ax=ax)
        ax.set(
            title="Skin Type Classifier — Confusion Matrix",
            xlabel="Predicted Label",
            ylabel="True Label",
            xticks=range(len(classes)),
            yticks=range(len(classes)),
        )
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        plt.tight_layout()
        out = OUTPUT_DIR / "confusion_matrix_skin.png"
        fig.savefig(out, dpi=120)
        plt.close(fig)
        ok(f"Confusion matrix saved → {out.name}")

    else:
        warn("Skipping Skin Model — missing model or text column")
        results["Skin Model"] = {"status": "skipped"}

except Exception as exc:
    err(f"Skin Model evaluation failed: {exc}")
    traceback.print_exc()
    results["Skin Model"] = {"status": "error", "detail": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — SENTIMENT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

sep("MODEL 2 — SENTIMENT ANALYSIS")

try:
    if sentiment_model is None:
        err("sentiment_model.pkl missing — skipping")
        results["Sentiment Model"] = {"status": "skipped"}
    else:
        mode = sentiment_model.get("mode", "vader") if isinstance(sentiment_model, dict) else "unknown"
        print(f"  Mode: {BOLD}{mode}{RESET}")

        # Locate review text
        text_col2  = find_col(df_reviews, ["review_text", "review", "text"])
        rating_col = find_col(df_reviews, ["rating_value", "rating", "stars",
                                           "review_score", "aggregate_rating"])

        if text_col2 is None:
            warn("No review text column — sentiment eval skipped")
            results["Sentiment Model"] = {"status": "skipped — no text col"}
        else:
            sample = df_reviews[[text_col2]].dropna()
            if rating_col:
                sample = df_reviews[[text_col2, rating_col]].dropna()

            sample = sample[sample[text_col2].astype(str).str.len() > 5]
            sample = sample.sample(min(3000, len(sample)), random_state=42)

            def vader_label(compound: float) -> str:
                if compound >= 0.05: return "positive"
                if compound <= -0.05: return "negative"
                return "neutral"

            if mode == "vader":
                analyzer = sentiment_model.get("analyzer")
                preds = sample[text_col2].astype(str).apply(
                    lambda t: vader_label(analyzer.polarity_scores(t)["compound"])
                )
            elif mode == "classifier":
                pipeline = sentiment_model.get("pipeline")
                preds = pd.Series(pipeline.predict(sample[text_col2].astype(str).tolist()))
            else:
                from textblob import TextBlob
                preds = sample[text_col2].astype(str).apply(
                    lambda t: "positive" if TextBlob(t).sentiment.polarity > 0.05
                    else ("negative" if TextBlob(t).sentiment.polarity < -0.05 else "neutral")
                )

            dist = preds.value_counts(normalize=True).round(3) * 100
            print(f"  Sentiment Distribution (n={len(preds):,}):")
            for label, pct in dist.items():
                print(f"    {label:12s}: {pct:.1f}%")

            if rating_col and rating_col in sample.columns:
                true_labels = sample[rating_col].apply(
                    lambda r: "positive" if r >= 4 else ("negative" if r <= 2 else "neutral")
                )
                preds_arr = preds.values if isinstance(preds, pd.Series) else preds
                true_arr  = true_labels.values

                from sklearn.metrics import accuracy_score as _acc
                proxy_acc = _acc(true_arr, preds_arr[:len(true_arr)])
                print(f"\n  Proxy Accuracy (vs ratings ≥4/≤2): {BOLD}{proxy_acc*100:.2f}%{RESET}")
                results["Sentiment Model"] = {
                    "mode": mode,
                    "Proxy Accuracy": f"{proxy_acc*100:.2f}%",
                    "Distribution": dist.to_dict(),
                }
            else:
                warn("No rating column — proxy accuracy unavailable")
                results["Sentiment Model"] = {"mode": mode, "Distribution": dist.to_dict()}

except Exception as exc:
    err(f"Sentiment evaluation failed: {exc}")
    traceback.print_exc()
    results["Sentiment Model"] = {"status": "error", "detail": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3 — RANDOM FOREST DEMAND FORECASTING
# ─────────────────────────────────────────────────────────────────────────────

sep("MODEL 3 — RANDOM FOREST DEMAND FORECAST")

try:
    if rf_model_payload is None:
        err("rf_forecast_model.pkl missing — skipping")
        results["RF Forecast"] = {"status": "skipped"}
    else:
        rf        = rf_model_payload.get("model") if isinstance(rf_model_payload, dict) else rf_model_payload
        feat_cols = (rf_model_payload.get("feature_cols", [])
                     if isinstance(rf_model_payload, dict) else [])

        # Detect columns
        qty_col  = find_col(df_sales, ["units_sold", "sales", "venda", "revenues", "estoque"])
        date_col = find_col(df_sales, ["start_date", "order_date", "date", "date_published"])

        if qty_col is None or date_col is None:
            warn(f"Sales qty/date cols not found (qty={qty_col}, date={date_col}) — using numeric fallback")
            # Fallback: use last 200 numeric rows
            num_cols = df_sales.select_dtypes(include="number").columns.tolist()
            if num_cols:
                qty_col = num_cols[0]
                df_sales["_fake_date"] = pd.date_range("2024-01-01", periods=len(df_sales), freq="D")
                date_col = "_fake_date"

        if qty_col and date_col:
            ts_df = df_sales[[date_col, qty_col]].copy()
            ts_df[date_col] = pd.to_datetime(ts_df[date_col], errors="coerce")
            ts_df[qty_col]  = pd.to_numeric(ts_df[qty_col], errors="coerce")
            ts_df = ts_df.dropna()
            ts_df = ts_df[ts_df[qty_col] >= 0]

            weekly = (
                ts_df.groupby(pd.Grouper(key=date_col, freq="W"))[qty_col]
                .sum()
                .sort_index()
            )

            print(f"  Weekly time-series: {len(weekly)} weeks  "
                  f"({weekly.index.min().date()} → {weekly.index.max().date()})")
            print(f"  Qty column used   : {qty_col}")
            print(f"  Model features    : {feat_cols or '(auto-detect)'}")

            # Engineer features matching training
            def build_features(ts: pd.Series) -> pd.DataFrame:
                df = pd.DataFrame({"y": ts})
                df["month"]          = df.index.month
                df["week_of_year"]   = df.index.isocalendar().week.astype(int)
                df["quarter"]        = df.index.quarter
                df["season"]         = (df["quarter"] - 1) % 4 + 1
                df["is_holiday"]     = (df["quarter"] == 4).astype(int)
                df["lag_1"]          = df["y"].shift(1)
                df["lag_2"]          = df["y"].shift(2)
                df["lag_4"]          = df["y"].shift(4)
                df["lag_8"]          = df["y"].shift(8)
                df["rolling_mean_4"] = df["y"].shift(1).rolling(4).mean()
                return df.dropna()

            feat_df   = build_features(weekly)
            all_fcols = [c for c in feat_df.columns if c != "y"]

            # Use only columns RF was trained on (if known)
            if feat_cols:
                use_cols = [c for c in feat_cols if c in feat_df.columns]
                if not use_cols:
                    use_cols = all_fcols
            else:
                use_cols = all_fcols

            X_eval = feat_df[use_cols].values
            y_eval = feat_df["y"].values

            split  = int(len(X_eval) * 0.8)
            X_test_rf = X_eval[split:]
            y_test_rf = y_eval[split:]

            if len(X_test_rf) < 2:
                warn("Not enough test samples for RF eval — using all data")
                X_test_rf = X_eval
                y_test_rf = y_eval

            y_pred_rf = rf.predict(X_test_rf)
            mae_rf    = float(np.mean(np.abs(y_test_rf - y_pred_rf)))
            rmse_rf   = float(np.sqrt(np.mean((y_test_rf - y_pred_rf) ** 2)))
            r2_rf     = 1 - np.sum((y_test_rf - y_pred_rf)**2) / (np.sum((y_test_rf - np.mean(y_test_rf))**2) + 1e-9)

            print(f"\n  MAE  : {BOLD}{mae_rf:.2f}{RESET} units/week")
            print(f"  RMSE : {BOLD}{rmse_rf:.2f}{RESET} units/week")
            print(f"  R²   : {r2_rf:.4f}")

            results["RF Forecast"] = {
                "MAE":  f"{mae_rf:.2f}",
                "RMSE": f"{rmse_rf:.2f}",
                "R2":   f"{r2_rf:.4f}",
            }

            # ── Forecast vs Actual Plot ────────────────────────────────────
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            ax1, ax2  = axes

            weeks = range(len(y_test_rf))
            ax1.plot(weeks, y_test_rf, label="Actual", color="#1a73e8", linewidth=2)
            ax1.plot(weeks, y_pred_rf, label="Predicted", color="#e84034",
                     linewidth=2, linestyle="--")
            ax1.set_title("RF Forecast — Actual vs Predicted")
            ax1.set_xlabel("Week (test set)")
            ax1.set_ylabel("Units Sold")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            errors = y_test_rf - y_pred_rf
            ax2.hist(errors, bins=30, color="#1a73e8", edgecolor="white", alpha=0.8)
            ax2.axvline(0, color="red", linestyle="--", linewidth=1.5)
            ax2.set_title("RF Forecast — Residual (Error) Distribution")
            ax2.set_xlabel("Prediction Error (Actual − Predicted)")
            ax2.set_ylabel("Frequency")
            ax2.grid(True, alpha=0.3)

            plt.suptitle("Random Forest Demand Forecast Evaluation", fontsize=13, fontweight="bold")
            plt.tight_layout()
            out2 = OUTPUT_DIR / "forecast_actual_vs_predicted.png"
            fig.savefig(out2, dpi=120)
            plt.close(fig)
            ok(f"Forecast plot saved → {out2.name}")

        else:
            warn("Could not build sales time series — RF eval skipped")
            results["RF Forecast"] = {"status": "skipped — no valid sales cols"}

except Exception as exc:
    err(f"RF Forecast evaluation failed: {exc}")
    traceback.print_exc()
    results["RF Forecast"] = {"status": "error", "detail": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 4 — ARIMA DEMAND FORECASTING
# ─────────────────────────────────────────────────────────────────────────────

sep("MODEL 4 — ARIMA DEMAND FORECASTING")

try:
    if arima_payload is None:
        err("arima_model.pkl missing — skipping")
        results["ARIMA"] = {"status": "skipped"}
    elif isinstance(arima_payload, dict) and "error" in arima_payload:
        warn(f"ARIMA model stored as error: {arima_payload['error']}")
        results["ARIMA"] = {"status": f"error in training: {arima_payload['error']}"}
    else:
        arima_models = arima_payload.get("models", {}) if isinstance(arima_payload, dict) else {}
        top_products = arima_payload.get("top_products", []) if isinstance(arima_payload, dict) else []

        print(f"  Products with ARIMA models: {len(arima_models)}")
        print(f"  Top products               : {top_products[:5]}")

        if arima_models:
            pid        = list(arima_models.keys())[0]
            fitted     = arima_models[pid]
            steps      = 4
            forecast   = fitted.forecast(steps=steps)
            print(f"\n  Sample forecast for product '{pid}' ({steps} weeks ahead):")
            for i, val in enumerate(forecast, 1):
                print(f"    Week +{i}: {val:.1f} units")
            results["ARIMA"] = {
                "n_models": len(arima_models),
                "sample_product": str(pid),
                "forecast_4wk": [round(float(v), 1) for v in forecast],
            }
        else:
            warn("No fitted ARIMA models found in pickle")
            results["ARIMA"] = {"status": "no models in payload"}

except Exception as exc:
    err(f"ARIMA evaluation failed: {exc}")
    traceback.print_exc()
    results["ARIMA"] = {"status": "error", "detail": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 5 — TF-IDF + COSINE SIMILARITY
# ─────────────────────────────────────────────────────────────────────────────

sep("MODEL 5 — TF-IDF INGREDIENT SIMILARITY")

try:
    if tfidf_vec is None or cosine_matrix is None:
        err("TF-IDF vectorizer or cosine matrix missing — skipping")
        results["Similarity"] = {"status": "skipped"}
    else:
        # Matrix shape
        shape = cosine_matrix.shape
        print(f"  TF-IDF matrix : {shape[0]:,} products × {tfidf_vec.get_feature_names_out().shape[0]:,} features")
        print(f"  Cosine matrix : {shape[0]:,} × {shape[1]:,}  (sparse: {hasattr(cosine_matrix, 'nnz')})")

        # Product index lookup
        product_ids   = id_index.get("product_ids",   []) if id_index else []
        product_names = id_index.get("product_names", []) if id_index else []
        names_map = dict(zip(product_ids, product_names))

        print(f"  Products indexed: {len(product_ids):,}")

        if product_ids:
            query_idx    = 0
            query_name   = product_names[query_idx] if product_names else "unknown"

            # Get row of similarity scores
            row   = cosine_matrix[query_idx]
            sims  = (row.toarray().flatten() if hasattr(row, "toarray") else np.asarray(row).flatten())
            top_n = np.argsort(sims)[::-1][1:6]  # skip self

            print(f"\n  Query product: '{query_name}'")
            print(f"  Top-5 similar products:")
            for rank, idx in enumerate(top_n, 1):
                name = product_names[idx] if idx < len(product_names) else str(product_ids[idx])
                sim  = sims[idx]
                print(f"    {rank}. {name[:55]:55s}  sim={sim:.4f}")

            # ── Similarity bar chart ───────────────────────────────────────
            sim_names  = [product_names[i][:30] if i < len(product_names)
                          else str(product_ids[i]) for i in top_n]
            sim_scores = [sims[i] for i in top_n]

            fig, ax = plt.subplots(figsize=(9, 5))
            bars = ax.barh(sim_names[::-1], sim_scores[::-1], color="#1a73e8")
            ax.set_xlabel("Cosine Similarity Score")
            ax.set_title(f"Top-5 Similar Products for:\n'{query_name[:60]}'")
            ax.bar_label(bars, fmt="%.3f", padding=3)
            ax.set_xlim(0, min(max(sim_scores) * 1.2, 1.05))
            ax.grid(True, alpha=0.3, axis="x")
            plt.tight_layout()
            out3 = OUTPUT_DIR / "similarity_top5.png"
            fig.savefig(out3, dpi=120)
            plt.close(fig)
            ok(f"Similarity chart saved → {out3.name}")

            results["Similarity"] = {
                "n_products":     len(product_ids),
                "n_tfidf_feats":  int(tfidf_vec.get_feature_names_out().shape[0]),
                "matrix_shape":   f"{shape[0]}×{shape[1]}",
                "sample_query":   query_name,
                "top_1_similar":  product_names[top_n[0]] if top_n.size else "n/a",
                "top_1_score":    f"{sims[top_n[0]]:.4f}" if top_n.size else "n/a",
            }

except Exception as exc:
    err(f"Similarity evaluation failed: {exc}")
    traceback.print_exc()
    results["Similarity"] = {"status": "error", "detail": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 6 — HARMFUL INGREDIENT DETECTOR
# ─────────────────────────────────────────────────────────────────────────────

sep("MODEL 6 — HARMFUL INGREDIENT DETECTOR")

try:
    if harmful_detector is None:
        err("harmful_detector.pkl missing — skipping")
        results["Harmful Detector"] = {"status": "skipped"}
    else:
        # Understand the payload
        if isinstance(harmful_detector, dict):
            rules  = harmful_detector.get("rules",           {})
            cats   = harmful_detector.get("severity_groups", {})
            hwords = harmful_detector.get("harmful_set",     set())
        elif isinstance(harmful_detector, (set, frozenset)):
            hwords = harmful_detector
            rules  = {}
        else:
            hwords = set()
            rules  = {}

        n_rules = len(rules) or len(hwords)
        print(f"  Harmful keywords/rules loaded: {n_rules}")

        # Test strings — escalating risk
        test_ingredients = [
            ("Safe product",     "Water, Glycerin, Aloe Vera, Vitamin E, Jojoba Oil"),
            ("Moderate product", "Water, Methylparaben, Sodium Laureth Sulfate, Fragrance"),
            ("Unsafe product",   "Formaldehyde, Lead Acetate, Mercury, Parabens, Triclosan, Phthalates"),
        ]

        print(f"\n  {'Test Product':<20} {'Ingredient Text':<50}  Detected")
        print("  " + "─" * 90)

        for label, ingr in test_ingredients:
            ingr_lower  = ingr.lower()
            if rules:
                detected = [r for r in rules if r in ingr_lower]
            else:
                detected = [w for w in hwords if w in ingr_lower]
            flag = f"{RED}FLAGGED ({len(detected)}){RESET}" if detected else f"{GREEN}CLEAN{RESET}"
            print(f"  {label:<20} {ingr[:48]:<50}  {flag}")
            if detected:
                print(f"    → Detected: {', '.join(detected[:5])}")

        results["Harmful Detector"] = {
            "n_keywords": n_rules,
            "test_cases":  3,
            "status": "operational",
        }

        # Also test against real dataset ingredients
        ingr_col = find_col(df_master, ["ingredients", "ingredient_list", "composition"])
        if ingr_col:
            sample_ingr = df_master[ingr_col].dropna().head(500).astype(str)
            keyword_list = list(rules.keys()) if rules else list(hwords)

            def count_harmful(text):
                t = text.lower()
                return sum(1 for kw in keyword_list if kw in t)

            df_master["_harmful_count"] = sample_ingr.apply(count_harmful).reindex(df_master.index, fill_value=0)
            n_flagged = (df_master["_harmful_count"] > 0).sum()
            print(f"\n  Real dataset check (first 500 products):")
            print(f"    Flagged products: {n_flagged} / 500 ({n_flagged/5:.1f}%)")

except Exception as exc:
    err(f"Harmful Detector evaluation failed: {exc}")
    traceback.print_exc()
    results["Harmful Detector"] = {"status": "error", "detail": str(exc)}


# ─────────────────────────────────────────────────────────────────────────────
# FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

sep("FINAL MODEL PERFORMANCE SUMMARY")

print(f"  {'Model':<25}  {'Key Metric':<20}  {'Value'}")
print("  " + "─" * 65)

def fmt_result(model_key, metric_key, fallback="N/A"):
    r = results.get(model_key, {})
    if "status" in r:
        return r["status"]
    return r.get(metric_key, fallback)

rows = [
    ("Skin Classifier",  "Accuracy",       fmt_result("Skin Model",       "Accuracy")),
    ("Sentiment Model",  "Proxy Accuracy", fmt_result("Sentiment Model",  "Proxy Accuracy")),
    ("RF Forecast",      "RMSE",           fmt_result("RF Forecast",      "RMSE") + " units" if "%" not in fmt_result("RF Forecast", "RMSE") else fmt_result("RF Forecast", "RMSE")),
    ("RF Forecast",      "MAE",            fmt_result("RF Forecast",      "MAE") + " units" if "%" not in fmt_result("RF Forecast", "MAE") else fmt_result("RF Forecast", "MAE")),
    ("ARIMA",            "Models fitted",  str(results.get("ARIMA", {}).get("n_models", "—"))),
    ("Similarity",       "Products",       str(results.get("Similarity", {}).get("n_products", "—"))),
    ("Similarity",       "TF-IDF feats",   str(results.get("Similarity", {}).get("n_tfidf_feats", "—"))),
    ("Harmful Detector", "Keywords",       str(results.get("Harmful Detector", {}).get("n_keywords", "—"))),
]

for model, metric, value in rows:
    status_icon = f"{GREEN}✔{RESET}" if value not in ("skipped", "error", "—", "N/A") else f"{YELLOW}⚠{RESET}"
    print(f"  {status_icon} {model:<24} {metric:<20} {value}")

print()
print(f"  {BOLD}Evaluation Outputs:{RESET}")
for f in sorted(OUTPUT_DIR.glob("*.png")):
    print(f"    → {f}")

print(f"\n  {GREEN}{BOLD}✅ Evaluation pipeline complete!{RESET}")
print(f"  All outputs saved to: {OUTPUT_DIR}\n")
