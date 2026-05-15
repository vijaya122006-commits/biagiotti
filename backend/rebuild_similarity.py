"""
rebuild_similarity.py
=====================
Rebuilds tfidf_vectorizer.pkl + cosine_similarity_matrix.pkl + product_id_index.pkl
from the live app database — compatible with the current numpy/sklearn environment.

Run from:  biagiotti/backend/
Command:   python rebuild_similarity.py
"""
import pickle, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

DB_PATH     = Path(__file__).parent.parent / "cosmetic_intel.db"
MODELS_DIR  = Path(__file__).parent / "models"

print("=" * 60)
print("  Biagiotti — Similarity Matrix Rebuilder")
print("=" * 60)

# ── 1. Load products from DB ──────────────────────────────────
import sqlite3
print(f"\n[1/4] Reading products from {DB_PATH.name}...")

if not DB_PATH.exists():
    print(f"  ERROR: Database not found at {DB_PATH}")
    sys.exit(1)

conn = sqlite3.connect(str(DB_PATH))
cur  = conn.cursor()

cur.execute("""
    SELECT product_id, product_name, brand, category, ingredients, skin_suitability
    FROM products
    LIMIT 5000
""")
rows = cur.fetchall()
conn.close()

if not rows:
    print("  ERROR: No products found in database. Run a sync first.")
    sys.exit(1)

product_ids   = []
product_names = []
corpus        = []

for row in rows:
    pid, name, brand, category, ingredients, skin_suitability = row
    if not name:
        continue

    # ── Weighted corpus: repeat brand 5× and category 3× ──────────────
    # This makes TF-IDF treat brand as the strongest signal,
    # so same-brand products cluster far more tightly.
    brand_w    = " ".join([str(brand or "")] * 5)
    category_w = " ".join([str(category or "")] * 3)

    text = " ".join(filter(None, [
        brand_w,
        category_w,
        str(name or ""),
        str(ingredients or "")[:300],
        str(skin_suitability or "")[:100],
    ]))
    product_ids.append(str(pid))
    product_names.append(str(name))
    corpus.append(text.lower())

print(f"  ✅  {len(corpus)} products loaded  (brand×5, category×3 weighting)")

# ── 2. TF-IDF Vectorizer ─────────────────────────────────────
print("\n[2/4] Fitting TF-IDF vectorizer...")
t0 = time.perf_counter()

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=12000,      # more vocab → better discrimination
    ngram_range=(1, 2),
    min_df=2,                # ignore terms that appear only once (noise)
    sublinear_tf=True,
    stop_words="english",
)
tfidf_matrix = tfidf.fit_transform(corpus)
print(f"  ✅  Matrix shape: {tfidf_matrix.shape}  ({time.perf_counter()-t0:.2f}s)")

# ── 3. Cosine Similarity ─────────────────────────────────────
print("\n[3/4] Computing cosine similarity matrix...")
t0 = time.perf_counter()

from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import numpy as np

# Compute full similarity then zero-out weak pairs (threshold = 0.20)
# This kills the 85% density problem and removes irrelevant recommendations.
THRESHOLD = 0.20
sim_full   = cosine_similarity(tfidf_matrix, dense_output=True)
np.fill_diagonal(sim_full, 0)                      # remove self-similarity
sim_full[sim_full < THRESHOLD] = 0                 # apply threshold
sim_sparse = sp.csr_matrix(sim_full)               # convert to sparse

density = sim_sparse.nnz / (len(corpus) ** 2) * 100
print(f"  ✅  Similarity matrix: {sim_sparse.shape}  ({time.perf_counter()-t0:.2f}s)")
print(f"       Threshold applied : sim >= {THRESHOLD}")
print(f"       Non-zero entries  : {sim_sparse.nnz:,}")
print(f"       Matrix density    : {density:.2f}%  (was 85.94% before)")

# ── 4. Save all three files ───────────────────────────────────
print("\n[4/4] Saving models...")
MODELS_DIR.mkdir(exist_ok=True)

with open(MODELS_DIR / "tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f, protocol=4)
print("  ✅  tfidf_vectorizer.pkl saved")

with open(MODELS_DIR / "cosine_similarity_matrix.pkl", "wb") as f:
    pickle.dump(sim_sparse, f, protocol=4)
print("  ✅  cosine_similarity_matrix.pkl saved")

id_index = {"product_ids": product_ids, "product_names": product_names}
with open(MODELS_DIR / "product_id_index.pkl", "wb") as f:
    pickle.dump(id_index, f, protocol=4)
print("  ✅  product_id_index.pkl saved")

# ── Quick sanity check ────────────────────────────────────────
print("\n── Quick sanity check ──")
scores = sim_sparse[0].toarray().flatten()
import numpy as np
top_idx = np.argsort(-scores)[1:4]
print(f"  Query: {product_names[0]}")
for i in top_idx:
    print(f"    → {product_names[i]:<50} sim={scores[i]:.4f}")

print("\n" + "=" * 60)
print("  ✅  Rebuild complete! Restart the Flask server to apply.")
print("=" * 60)
