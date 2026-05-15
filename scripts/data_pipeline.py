"""
=============================================================
  COSMETIC DATA PIPELINE
  Full ingestion -> cleaning -> merging -> verified_products -> ML-ready
=============================================================
"""
import os, sys, re, zipfile, warnings, random
from pathlib import Path
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# Force UTF-8 output
sys.stdout.reconfigure(encoding='utf-8')

# ---- Paths ------------------------------------------------------------------
# Anchored to biagiotti/backend/data/ (scripts/ is one level below project root)
_RAW_DATA    = Path(__file__).resolve().parent.parent / "backend" / "data" / "raw"
INGR_DIR     = _RAW_DATA / "ingredients"
PROD_DIR     = _RAW_DATA / "products"
REV_DIR      = _RAW_DATA / "reviews"
SALES_DIR    = _RAW_DATA / "sales"
VERIFIED_DIR = Path(__file__).resolve().parent.parent / "backend" / "data" / "verified_products"
CLEANED_DIR  = Path(__file__).resolve().parent.parent / "backend" / "data" / "cleaned"
VERIFIED_DIR.mkdir(exist_ok=True)
CLEANED_DIR.mkdir(exist_ok=True)

SEP = "=" * 70

# =============================================================================
# UTILITY HELPERS
# =============================================================================

def clean_col_names(df):
    df.columns = (
        df.columns.str.strip()
          .str.lower()
          .str.replace(r"[\s\-/]+", "_", regex=True)
          .str.replace(r"[^\w]", "", regex=True)
    )
    return df


def safe_read_csv(path, **kwargs):
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            df = pd.read_csv(path, encoding=enc, on_bad_lines="skip", **kwargs)
            return df
        except Exception:
            continue
    print(f"  [WARN] FAILED to read {path}")
    return pd.DataFrame()


def unzip_all(folder):
    for z in folder.glob("*.zip"):
        print(f"  [ZIP] Unzipping {z.name} ...")
        try:
            with zipfile.ZipFile(z, "r") as zf:
                zf.extractall(folder)
        except Exception as e:
            print(f"  [WARN] Could not unzip {z.name}: {e}")


def load_folder(folder, label):
    unzip_all(folder)
    frames = []
    for csv in sorted(folder.glob("**/*.csv")):
        df = safe_read_csv(str(csv))
        if df.empty:
            print(f"  [WARN] Empty/unreadable: {csv.name}")
            continue
        df = clean_col_names(df)
        df["_source_file"] = csv.name
        print(f"  [OK] {label}/{csv.name}  -> {df.shape[0]:,} rows x {df.shape[1]} cols")
        print(f"       columns: {list(df.columns)}")
        frames.append(df)
    return frames


def normalize_text(series):
    return series.astype(str).str.strip().str.lower().str.replace(r"\s+", " ", regex=True)


def make_id(series, prefix):
    cats = pd.Categorical(normalize_text(series))
    mapping = {v: f"{prefix}_{i:05d}" for i, v in enumerate(cats.categories)}
    return normalize_text(series).map(mapping)


def get_season(date_series):
    def _s(m):
        if m in (12, 1, 2):    return "winter"
        if m in (3, 4, 5):    return "spring"
        if m in (6, 7, 8, 9): return "rainy"
        return "autumn"
    months = pd.to_datetime(date_series, errors="coerce").dt.month
    return months.map(_s).fillna("unknown")


def random_expiry(n):
    base = datetime(2026, 4, 1)
    return [
        (base + timedelta(days=random.randint(365, 1095))).strftime("%Y-%m")
        for _ in range(n)
    ]


HARMFUL_INGREDIENTS = {
    "parabens","formaldehyde","phthalates","lead","mercury",
    "triclosan","oxybenzone","coal tar","bha","bht",
    "petroleum","sodium lauryl sulfate","sls",
}


def safety_score(ingredient_text):
    def _score(txt):
        txt = str(txt).lower()
        hits = sum(1 for h in HARMFUL_INGREDIENTS if h in txt)
        return max(0, 10 - hits)
    return ingredient_text.apply(_score)


OILY_TERMS      = {"oily","shine","sebum","grease","pore","pores"}
DRY_TERMS       = {"dry","flaky","tight","rough","dehydrated"}
SENSITIVE_TERMS = {"sensitive","irritation","redness","allergy","sting"}


def skin_suitability(review_text):
    def _suit(txt):
        txt = str(txt).lower()
        scores = {
            "oily":      sum(w in txt for w in OILY_TERMS),
            "dry":       sum(w in txt for w in DRY_TERMS),
            "sensitive": sum(w in txt for w in SENSITIVE_TERMS),
        }
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else "normal"
    return review_text.apply(_suit)


STOPWORDS = {
    "i","me","my","the","a","an","and","or","but","is","was","are","were",
    "it","its","this","that","to","of","in","on","for","with","at","by",
    "from","as","be","been","being","have","has","had","do","does","did",
    "will","would","could","should","may","might","not","no","so","we","they",
    "he","she","you","your","our","their","his","her","if","then","there",
    "when","where","which","who","what","how","all","just","also","very",
}


def tokenize_clean(series):
    def _tok(txt):
        tokens = re.findall(r"[a-zA-Z]+", str(txt).lower())
        return " ".join(t for t in tokens if t not in STOPWORDS and len(t) > 2)
    return series.apply(_tok)


PRODUCT_NAME_ALIASES = ["product_name","name","productname","product","title",
                        "label","item","item_name","product_title"]
BRAND_ALIASES        = ["brand","brand_name","brandname","manufacturer","company"]
PRICE_ALIASES        = ["price","price_usd","cost","amount","sale_price","list_price"]
RATING_ALIASES       = ["rating","ratings","avg_rating","average_rating","stars",
                        "review_score","score"]
INGREDIENT_ALIASES   = ["ingredients","ingredient","ingredient_list",
                        "chemicals","chemical_ingredients","composition"]
REVIEW_ALIASES       = ["review","review_text","reviews","comment","feedback",
                        "text","review_body","review_comment","cleaned_review"]


def first_match(df, aliases):
    for a in aliases:
        if a in df.columns:
            return a
    return None


def unify_col(df, aliases, target):
    m = first_match(df, aliases)
    if m and m != target:
        df = df.rename(columns={m: target})
    return df


def standardise(df):
    df = df.copy()
    for aliases, target in [
        (PRODUCT_NAME_ALIASES, "product_name"),
        (BRAND_ALIASES,        "brand"),
        (PRICE_ALIASES,        "price"),
        (RATING_ALIASES,       "rating"),
        (INGREDIENT_ALIASES,   "ingredients"),
        (REVIEW_ALIASES,       "review_text"),
    ]:
        df = unify_col(df, aliases, target)

    for col in ["product_name", "brand", "shade"]:
        if col in df.columns:
            df[col] = normalize_text(df[col])

    if "product_name" in df.columns:
        df["product_id"] = make_id(df["product_name"], "PRD")
    if "brand" in df.columns:
        df["brand_id"] = make_id(df["brand"], "BRD")
    if "shade" in df.columns:
        df["shade_id"] = make_id(df["shade"], "SHD")

    if "price" in df.columns:
        df["price"] = pd.to_numeric(
            df["price"].astype(str).str.replace(r"[^\d\.]", "", regex=True),
            errors="coerce"
        )

    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        mx = df["rating"].dropna().max()
        if mx and mx > 5:
            df["rating"] = df["rating"] / mx * 5

    before = len(df)
    df = df.drop_duplicates()
    dropped = before - len(df)
    if dropped:
        print(f"    [DUP] Removed {dropped:,} duplicate rows")

    for col in df.select_dtypes(include="number").columns:
        df[col] = df[col].fillna(df[col].median())
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].fillna("unknown")

    return df


def smart_concat(frames, label):
    if not frames:
        print(f"  [WARN] No frames for {label}")
        return pd.DataFrame()
    merged = pd.concat(frames, ignore_index=True, sort=False)
    merged = merged.drop_duplicates()
    print(f"  [OK] {label}: {merged.shape[0]:,} rows x {merged.shape[1]} cols")
    return merged


def product_ids_in(df):
    if "product_id" in df.columns:
        return set(df["product_id"].dropna().unique())
    if "product_name" in df.columns:
        return set(make_id(df["product_name"], "PRD").dropna().unique())
    return set()


def save_cleaned(df, name):
    if df.empty:
        print(f"  [WARN] {name} is empty -- skipping.")
        return
    df_clean = df.drop(columns=[c for c in ["_source_file"] if c in df.columns])
    path = CLEANED_DIR / name
    df_clean.to_csv(path, index=False, encoding="utf-8")
    print(f"  [SAVED] {name}: {df_clean.shape[0]:,} rows x {df_clean.shape[1]} cols")


# =============================================================================
# STEP 1 - READ ALL DATASETS
# =============================================================================
print(f"\n{SEP}")
print("STEP 1 - READING ALL DATASETS")
print(SEP)

print("\n[DIR] INGREDIENTS ...")
ingr_frames = load_folder(INGR_DIR, "ingredients")

print("\n[DIR] PRODUCTS ...")
prod_frames = load_folder(PROD_DIR, "products")

print("\n[DIR] REVIEWS ...")
rev_frames = load_folder(REV_DIR, "reviews")

print("\n[DIR] SALES ...")
sales_frames = load_folder(SALES_DIR, "sales")

# =============================================================================
# STEP 2 - STANDARDISE
# =============================================================================
print(f"\n{SEP}")
print("STEP 2 - STANDARDISING DATASETS")
print(SEP)

ingr_std  = [standardise(df) for df in ingr_frames]
prod_std  = [standardise(df) for df in prod_frames]
rev_std   = [standardise(df) for df in rev_frames]
sales_std = [standardise(df) for df in sales_frames]
print("  [OK] Standardisation complete.")

# =============================================================================
# STEP 3 - MERGE INTO MASTER TABLES
# =============================================================================
print(f"\n{SEP}")
print("STEP 3 - MERGING INTO MASTER TABLES")
print(SEP)

master_ingredients = smart_concat(ingr_std,  "master_ingredients")
master_products    = smart_concat(prod_std,   "master_products")
master_reviews     = smart_concat(rev_std,    "master_reviews")
master_sales       = smart_concat(sales_std,  "master_sales")

# =============================================================================
# STEP 4 - VERIFIED PRODUCTS
# =============================================================================
print(f"\n{SEP}")
print("STEP 4 - CREATING VERIFIED PRODUCTS")
print(SEP)

ids_prod  = product_ids_in(master_products)
ids_ingr  = product_ids_in(master_ingredients)
ids_rev   = product_ids_in(master_reviews)
ids_sales = product_ids_in(master_sales)

verified_ids = ids_prod & (ids_ingr | ids_rev) & (ids_rev | ids_sales)
if len(verified_ids) < 10:
    verified_ids = (ids_prod & ids_rev) | (ids_prod & ids_sales) | (ids_prod & ids_ingr)
if len(verified_ids) < 5:
    verified_ids = ids_prod if ids_prod else set()

print(f"  [INFO] Verified product IDs found: {len(verified_ids):,}")

if "product_id" in master_products.columns and verified_ids:
    vp = master_products[master_products["product_id"].isin(verified_ids)].copy()
else:
    vp = master_products.copy()

if vp.empty:
    vp = master_products.copy()
    print("  [INFO] No strict cross-source matches; using all master_products as base.")

# Review aggregation
if not master_reviews.empty and "product_id" in master_reviews.columns:
    rev_agg = master_reviews.groupby("product_id").size().reset_index(name="review_count")
    if "rating" in master_reviews.columns:
        rat_agg = master_reviews.groupby("product_id")["rating"].mean().reset_index(name="average_rating")
        rat_agg["average_rating"] = rat_agg["average_rating"].round(2)
        rev_agg = rev_agg.merge(rat_agg, on="product_id", how="left")
    else:
        rev_agg["average_rating"] = 0.0
    # Drop any rev cols already in vp to avoid duplication
    merge_cols = [c for c in rev_agg.columns if c == "product_id" or c not in vp.columns]
    vp = vp.merge(rev_agg[merge_cols], on="product_id", how="left")
else:
    vp["review_count"]   = 0
    vp["average_rating"] = 0.0

# Sales aggregation
if not master_sales.empty:
    sales_col = next(
        (c for c in ["quantity","units_sold","sales","quantity_sold","qty"]
         if c in master_sales.columns), None
    )
    if sales_col and "product_id" in master_sales.columns:
        s_agg = (
            master_sales.groupby("product_id")[sales_col]
            .sum().reset_index()
            .rename(columns={sales_col: "total_sales"})
        )
        if "total_sales" not in vp.columns:
            vp = vp.merge(s_agg, on="product_id", how="left")
        else:
            vp["total_sales"] = vp["total_sales"].fillna(0)
    else:
        if "total_sales" not in vp.columns:
            vp["total_sales"] = 0
else:
    if "total_sales" not in vp.columns:
        vp["total_sales"] = 0

# Ingredients aggregation
if not master_ingredients.empty and "product_id" in master_ingredients.columns and "ingredients" in master_ingredients.columns:
    ingr_agg = (
        master_ingredients.groupby("product_id")["ingredients"]
        .apply(lambda x: "; ".join(x.dropna().astype(str).unique()))
        .reset_index(name="ingredients")
    )
    if "ingredients" not in vp.columns:
        vp = vp.merge(ingr_agg[["product_id","ingredients"]], on="product_id", how="left")

# Skin suitability
if "review_text" in vp.columns:
    vp["skin_suitability"] = skin_suitability(vp["review_text"])
elif not master_reviews.empty and "review_text" in master_reviews.columns and "product_id" in master_reviews.columns:
    suit_agg = (
        master_reviews.groupby("product_id")["review_text"]
        .apply(lambda x: " ".join(x.dropna().astype(str)))
        .reset_index(name="review_text")
    )
    suit_agg["skin_suitability"] = skin_suitability(suit_agg["review_text"])
    if "skin_suitability" not in vp.columns:
        vp = vp.merge(suit_agg[["product_id","skin_suitability"]], on="product_id", how="left")
else:
    vp["skin_suitability"] = "normal"

vp["verified_flag"] = 1

monk_col = next(
    (c for c in ["monk_shade","monk","monk_category","monk_skin_tone","skin_tone"] if c in vp.columns),
    None
)
if monk_col and monk_col != "monk_category":
    vp = vp.rename(columns={monk_col: "monk_category"})
elif not monk_col:
    vp["monk_category"] = "unknown"

KEEP_COLS = [
    "product_id","product_name","brand","ingredients","price","shade",
    "verified_flag","review_count","average_rating",
    "total_sales","monk_category","skin_suitability",
]
for c in KEEP_COLS:
    if c not in vp.columns:
        vp[c] = "unknown" if c in ("product_id","product_name","brand","ingredients",
                                    "shade","monk_category","skin_suitability") else 0

vp_final = vp[[c for c in KEEP_COLS if c in vp.columns]].copy()
vp_final[["review_count","total_sales"]] = vp_final[["review_count","total_sales"]].fillna(0)
vp_final["average_rating"] = vp_final.get("average_rating", pd.Series(dtype=float)).fillna(0.0)

if "product_id" in vp_final.columns:
    vp_final = vp_final.drop_duplicates(subset=["product_id"])

out_vp = VERIFIED_DIR / "verified_products.csv"
vp_final.to_csv(out_vp, index=False, encoding="utf-8")
print(f"  [SAVED] {out_vp}  ({vp_final.shape[0]:,} rows x {vp_final.shape[1]} cols)")

# =============================================================================
# STEP 5 - ML-READY MASTER DATASET
# =============================================================================
print(f"\n{SEP}")
print("STEP 5 - BUILDING ML-READY MASTER DATASET")
print(SEP)

ml = vp_final.copy()

# Season
if not master_sales.empty:
    date_col = next(
        (c for c in ["date","order_date","sale_date","transaction_date","created_at"]
         if c in master_sales.columns), None
    )
    if date_col and "product_id" in master_sales.columns:
        s_date = master_sales[["product_id", date_col]].dropna()
        s_date = s_date.copy()
        s_date["season"] = get_season(s_date[date_col])
        s_date_agg = (
            s_date.groupby("product_id")["season"]
            .agg(lambda x: x.mode().iloc[0] if len(x) else "unknown")
            .reset_index()
        )
        ml = ml.merge(s_date_agg, on="product_id", how="left")
    else:
        ml["season"] = "unknown"
else:
    ml["season"] = "unknown"

ml["season"] = ml["season"].fillna("unknown")

# Synthetic expiry
ml["expiry_month"] = random_expiry(len(ml))

# Popularity score
for c in ["review_count","total_sales"]:
    if c in ml.columns:
        ml[c] = pd.to_numeric(ml[c], errors="coerce").fillna(0)

rev_w   = ml["review_count"].astype(float) if "review_count" in ml.columns else pd.Series(0.0, index=ml.index)
sale_w  = ml["total_sales"].astype(float)  if "total_sales"  in ml.columns else pd.Series(0.0, index=ml.index)
max_rev  = max(rev_w.max(), 1)
max_sale = max(sale_w.max(), 1)
ml["popularity_score"] = ((0.4 * rev_w / max_rev) + (0.6 * sale_w / max_sale)).round(4)

# Safety score
ingr_col = "ingredients" if "ingredients" in ml.columns else None
ml["safety_score"] = safety_score(ml[ingr_col]) if ingr_col else 10

# Text cleaning
for col in ["product_name","brand","ingredients"]:
    if col in ml.columns:
        ml[f"{col}_clean"] = tokenize_clean(ml[col])

# Label encode categoricals
for col in ["skin_suitability","season","monk_category","brand","verified_flag"]:
    if col in ml.columns:
        ml[col] = ml[col].astype(str).fillna("unknown")
        uniques  = sorted(ml[col].unique(), key=lambda x: str(x))
        enc_map  = {v: i for i, v in enumerate(uniques)}
        ml[f"{col}_enc"] = ml[col].map(enc_map)

ml = ml.drop(columns=[c for c in ["_source_file"] if c in ml.columns])

out_ml = CLEANED_DIR / "master_cleaned.csv"
ml.to_csv(out_ml, index=False, encoding="utf-8")
print(f"  [SAVED] {out_ml}  ({ml.shape[0]:,} rows x {ml.shape[1]} cols)")

# =============================================================================
# STEP 6 - SAVE CLEANED SUB-TABLES
# =============================================================================
print(f"\n{SEP}")
print("STEP 6 - SAVING CLEANED DATASETS")
print(SEP)

save_cleaned(master_products,    "master_products_cleaned.csv")
save_cleaned(master_ingredients, "master_ingredients_cleaned.csv")
save_cleaned(master_reviews,     "master_reviews_cleaned.csv")
save_cleaned(master_sales,       "master_sales_cleaned.csv")

# =============================================================================
# STEP 7 - FULL SUMMARY REPORT
# =============================================================================
print(f"\n{SEP}")
print("STEP 7 - FINAL SUMMARY REPORT")
print(SEP)

report_data = {
    "master_cleaned.csv":             ml,
    "master_products_cleaned.csv":    master_products,
    "master_ingredients_cleaned.csv": master_ingredients,
    "master_reviews_cleaned.csv":     master_reviews,
    "master_sales_cleaned.csv":       master_sales,
    "verified_products.csv":          vp_final,
}

print(f"\n{'FILE':<40} {'ROWS':>8} {'COLS':>6} {'MISSING':>10} {'MISSING%':>10} {'STATUS':>12}")
print("-" * 88)
for fname, df in report_data.items():
    if df.empty:
        print(f"{fname:<40} {'0':>8} {'0':>6} {'0':>10} {'0.00%':>10} {'EMPTY':>12}")
        continue
    total_cells   = df.shape[0] * df.shape[1]
    missing_cells = int(df.isnull().sum().sum())
    missing_pct   = f"{missing_cells / total_cells * 100:.2f}%" if total_cells else "0%"
    status        = "ML-READY" if missing_cells == 0 else "MINOR GAPS"
    print(f"{fname:<40} {df.shape[0]:>8,} {df.shape[1]:>6} {missing_cells:>10,} {missing_pct:>10} {status:>12}")

# Per-column detail for ML master
print(f"\nMissing value detail -- master_cleaned.csv:")
miss = ml.isnull().sum()
miss = miss[miss > 0]
if miss.empty:
    print("   No missing values! Dataset is fully clean.")
else:
    for col, cnt in miss.items():
        print(f"   {col}: {cnt:,} missing ({cnt/len(ml)*100:.1f}%)")

print(f"\n{SEP}")
print("PIPELINE COMPLETE -- All outputs saved:")
print(f"  Verified : {VERIFIED_DIR}")
print(f"  Cleaned  : {CLEANED_DIR}")
print(SEP)
