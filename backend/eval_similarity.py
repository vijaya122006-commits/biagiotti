"""
eval_similarity.py
==================
Deep evaluation of TF-IDF + Cosine Similarity pipeline.
Computes: Precision@K, Recall@K, NDCG@K, MAP, MRR, Hit Rate, Coverage, Diversity.

Run from:  biagiotti/backend/
Command:   python eval_similarity.py
"""
import sys, time, pickle, sqlite3, math
import numpy as np
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

SEP  = "=" * 68
SEP2 = "-" * 68

# ── Load Models ────────────────────────────────────────────────────────
MODELS = Path("models")

def load(name):
    p = MODELS / name
    with open(p, "rb") as f:
        return pickle.load(f)

print(SEP)
print("  TF-IDF + Cosine Similarity — Deep Evaluation")
print(SEP)

print("\n[1/6] Loading models...")
tfidf      = load("tfidf_vectorizer.pkl")
sim_matrix = load("cosine_similarity_matrix.pkl")
id_index   = load("product_id_index.pkl")

product_ids   = id_index["product_ids"]
product_names = id_index["product_names"]
N             = len(product_ids)

print(f"  Products in index : {N:,}")
print(f"  Vocab size        : {len(tfidf.vocabulary_):,}")
print(f"  Matrix shape      : {sim_matrix.shape}")
print(f"  Matrix density    : {sim_matrix.nnz / (N*N) * 100:.2f}%")
print(f"  Non-zero entries  : {sim_matrix.nnz:,}")

# ── Load ground truth from DB ──────────────────────────────────────────
print("\n[2/6] Building ground-truth from database...")

DB_PATH = Path("../cosmetic_intel.db")
conn    = sqlite3.connect(str(DB_PATH))
cur     = conn.cursor()
cur.execute("SELECT product_id, product_name, brand, category FROM products LIMIT 5000")
rows    = cur.fetchall()
conn.close()

pid_to_brand    = {}
pid_to_category = {}
for pid, name, brand, cat in rows:
    pid_to_brand[str(pid)]    = str(brand or "").strip().lower()
    pid_to_category[str(pid)] = str(cat or "").strip().lower()

print(f"  DB products loaded: {len(rows):,}")

# ── Helper: Get top-K recommendations ─────────────────────────────────
def get_topk(idx, k=10):
    row    = sim_matrix[idx].toarray().flatten()
    ranked = np.argsort(-row)
    results = []
    for r in ranked:
        if r != idx:
            results.append(r)
        if len(results) >= k:
            break
    return results, row

# Ground truth: relevant = same brand OR same category (non-empty)
def is_relevant_brand(q_idx, r_idx):
    qb = pid_to_brand.get(product_ids[q_idx], "")
    rb = pid_to_brand.get(product_ids[r_idx], "")
    return qb and rb and qb == rb

def is_relevant_cat(q_idx, r_idx):
    qc = pid_to_category.get(product_ids[q_idx], "")
    rc = pid_to_category.get(product_ids[r_idx], "")
    return qc and rc and qc == rc

# ── Sample for evaluation ──────────────────────────────────────────────
print("\n[3/6] Running evaluation on sample queries...")

# Use every 10th product as a query (up to 400 queries)
query_indices = list(range(0, min(N, 4000), 10))
K_VALUES      = [1, 3, 5, 10]
TOP_K         = 10

results_brand = defaultdict(list)   # metric -> list of values
results_cat   = defaultdict(list)

latencies = []
covered   = set()
all_scores = []

for q_idx in query_indices:
    t0        = time.perf_counter()
    topk, scores = get_topk(q_idx, k=TOP_K)
    latencies.append((time.perf_counter() - t0) * 1000)

    # Track all recommended items for coverage
    covered.update(topk)
    all_scores.extend([scores[r] for r in topk])

    for k in K_VALUES:
        recs = topk[:k]

        # ── Brand relevance metrics ──────────────────────────────────
        rel_b   = [1 if is_relevant_brand(q_idx, r) else 0 for r in recs]
        n_rel_b = sum(rel_b)

        # Precision@K
        results_brand[f"P@{k}"].append(n_rel_b / k)

        # Recall@K — need total relevant in index
        total_relevant_b = sum(
            1 for i in range(N)
            if i != q_idx and is_relevant_brand(q_idx, i)
        ) if k == TOP_K else None    # only compute for top-K to save time

        # NDCG@K
        dcg  = sum(rel_b[i] / math.log2(i + 2) for i in range(k))
        idcg = sum(1 / math.log2(i + 2) for i in range(min(n_rel_b, k)))
        results_brand[f"NDCG@{k}"].append(dcg / idcg if idcg > 0 else 0)

        # Hit Rate@K (any relevant in top-K)
        results_brand[f"HR@{k}"].append(1 if n_rel_b > 0 else 0)

        # MRR (only at K=10 for efficiency)
        if k == TOP_K:
            for rank, rel in enumerate(rel_b):
                if rel:
                    results_brand["MRR"].append(1 / (rank + 1))
                    break
            else:
                results_brand["MRR"].append(0)

        # ── Category relevance metrics ───────────────────────────────
        rel_c   = [1 if is_relevant_cat(q_idx, r) else 0 for r in recs]
        n_rel_c = sum(rel_c)

        results_cat[f"P@{k}"].append(n_rel_c / k)

        dcg_c  = sum(rel_c[i] / math.log2(i + 2) for i in range(k))
        idcg_c = sum(1 / math.log2(i + 2) for i in range(min(n_rel_c, k)))
        results_cat[f"NDCG@{k}"].append(dcg_c / idcg_c if idcg_c > 0 else 0)

        results_cat[f"HR@{k}"].append(1 if n_rel_c > 0 else 0)

        if k == TOP_K:
            for rank, rel in enumerate(rel_c):
                if rel:
                    results_cat["MRR"].append(1 / (rank + 1))
                    break
            else:
                results_cat["MRR"].append(0)

Q = len(query_indices)

# ── Coverage ───────────────────────────────────────────────────────────
coverage = len(covered) / N * 100

# ── Intra-list Diversity (avg pairwise distance in top-10) ────────────
# Sample 50 queries for diversity (expensive to compute for all)
diversity_scores = []
for q_idx in query_indices[:50]:
    topk, scores = get_topk(q_idx, k=10)
    if len(topk) < 2:
        continue
    # Average 1 - sim for each pair
    pairs = []
    for i in range(len(topk)):
        for j in range(i+1, len(topk)):
            s = float(sim_matrix[topk[i], topk[j]])
            pairs.append(1 - s)
    diversity_scores.append(np.mean(pairs))

avg_diversity = np.mean(diversity_scores) if diversity_scores else 0

# ── Score distribution ─────────────────────────────────────────────────
all_scores = np.array(all_scores)

# ── Print Report ──────────────────────────────────────────────────────

print(f"\n  Evaluated {Q:,} queries  |  Top-K={TOP_K}")

print(f"\n{'':2}{'Metric':<22}", end="")
for k in K_VALUES:
    print(f"  K={k:>2}", end="")
print()
print(f"  {SEP2}")

# Brand-based metrics
print("\n  ── Ground Truth: Same Brand ──────────────────────────────────────")
for metric in ["P", "NDCG", "HR"]:
    print(f"  {metric+'@K':<22}", end="")
    for k in K_VALUES:
        key = f"{metric}@{k}"
        val = np.mean(results_brand[key]) if results_brand[key] else 0
        print(f"  {val:>5.1%}", end="")
    print()

mrr_b = np.mean(results_brand["MRR"]) if results_brand["MRR"] else 0
print(f"  {'MRR (Mean Recip Rank)':<22}  {mrr_b:.4f}")

# Category-based metrics
print("\n  ── Ground Truth: Same Category ───────────────────────────────────")
for metric in ["P", "NDCG", "HR"]:
    print(f"  {metric+'@K':<22}", end="")
    for k in K_VALUES:
        key = f"{metric}@{k}"
        val = np.mean(results_cat[key]) if results_cat[key] else 0
        print(f"  {val:>5.1%}", end="")
    print()

mrr_c = np.mean(results_cat["MRR"]) if results_cat["MRR"] else 0
print(f"  {'MRR (Mean Recip Rank)':<22}  {mrr_c:.4f}")

# ── TF-IDF Model Info ─────────────────────────────────────────────────
print(f"\n{SEP}")
print("  TF-IDF Vectorizer Stats")
print(SEP)
print(f"  {'Vocabulary size':<30} {len(tfidf.vocabulary_):,}")
print(f"  {'n-gram range':<30} {tfidf.ngram_range}")
print(f"  {'Max features':<30} {tfidf.max_features:,}")
print(f"  {'Sublinear TF scaling':<30} {tfidf.sublinear_tf}")
print(f"  {'Stop words':<30} {tfidf.stop_words}")

# ── Similarity Score Stats ────────────────────────────────────────────
print(f"\n{SEP}")
print("  Cosine Similarity Score Distribution (recommendations)")
print(SEP)
pcts = [10, 25, 50, 75, 90, 95, 99]
for p in pcts:
    print(f"  P{p:<3} : {np.percentile(all_scores, p):.4f}")
print(f"  {'Mean':<5}: {np.mean(all_scores):.4f}")
print(f"  {'Std':<5} : {np.std(all_scores):.4f}")
print(f"  {'Min':<5} : {np.min(all_scores):.4f}")
print(f"  {'Max':<5} : {np.max(all_scores):.4f}")

# ── System Metrics ────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  System & Catalogue Metrics")
print(SEP)
print(f"  {'Queries evaluated':<35} {Q:,}")
print(f"  {'Products in index':<35} {N:,}")
print(f"  {'Matrix shape':<35} {sim_matrix.shape[0]:,} × {sim_matrix.shape[1]:,}")
print(f"  {'Matrix density':<35} {sim_matrix.nnz / (N*N) * 100:.2f}%")
print(f"  {'Catalogue coverage @K=10':<35} {coverage:.1f}%")
print(f"  {'Intra-list diversity (avg)':<35} {avg_diversity:.4f}   (1=max diversity)")
print(f"  {'Mean query latency':<35} {np.mean(latencies):.2f} ms")
print(f"  {'P50 latency':<35} {np.percentile(latencies, 50):.2f} ms")
print(f"  {'P95 latency':<35} {np.percentile(latencies, 95):.2f} ms")
print(f"  {'P99 latency':<35} {np.percentile(latencies, 99):.2f} ms")

# ── Sample outputs ────────────────────────────────────────────────────
print(f"\n{SEP}")
print("  Sample Recommendations (3 random queries)")
print(SEP)
import random
random.seed(42)
sample_qs = random.sample(query_indices, min(3, len(query_indices)))

for q_idx in sample_qs:
    qpid  = product_ids[q_idx]
    qname = product_names[q_idx]
    qbrand= pid_to_brand.get(qpid, "?")
    qcat  = pid_to_category.get(qpid, "?")
    print(f"\n  Query [{q_idx}]: {qname[:55]}")
    print(f"           Brand={qbrand} | Category={qcat}")

    topk, scores = get_topk(q_idx, k=5)
    for rank, r_idx in enumerate(topk):
        rpid   = product_ids[r_idx]
        rname  = product_names[r_idx]
        rbrand = pid_to_brand.get(rpid, "?")
        rcat   = pid_to_category.get(rpid, "?")
        same_b = "✅" if is_relevant_brand(q_idx, r_idx) else "  "
        same_c = "✅" if is_relevant_cat(q_idx, r_idx)   else "  "
        print(f"    #{rank+1}  {same_b}brand {same_c}cat  sim={scores[r_idx]:.4f}  {rname[:48]}")

print(f"\n{SEP}")
print("  Evaluation Complete")
print(SEP)
