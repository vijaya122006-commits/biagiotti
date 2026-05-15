"""
============================================================
  TRAIN IMPROVED SKIN CLASSIFIER (v2)
  Advanced NLP + Class Balancing + GridSearchCV
============================================================
"""
import os
import re
import pickle
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import RandomOverSampler

# --- Configuration ---
DATA_PATH = "data/cleaned/master_reviews_cleaned.csv"
MASTER_PATH = "data/cleaned/master_cleaned.csv"
MODEL_PATH = "models/skin_model_v2.pkl"
VEC_PATH = "models/vectorizer_v2.pkl"
LOG_DIR = "evaluation_outputs"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs("models", exist_ok=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", "", text)  # Punctuation & Numbers
    text = re.sub(r"\s+", " ", text).strip() # Extra spaces
    return text

def run_pipeline():
    print("\n[1/7] Loading and Merging Datasets...")
    try:
        df_rev = pd.read_csv(DATA_PATH)
        df_master = pd.read_csv(MASTER_PATH)
    except Exception as e:
        print(f"Error loading datasets: {e}")
        return

    # Auto-detect columns
    text_col = "review_text" if "review_text" in df_rev.columns else None
    id_col = "product_id" if "product_id" in df_rev.columns and "product_id" in df_master.columns else None
    label_col = "skin_suitability" if "skin_suitability" in df_master.columns else None

    if not all([text_col, id_col, label_col]):
        print("Required columns missing for join.")
        return

    # Merge to get labels
    df = df_rev.merge(df_master[[id_col, label_col]], on=id_col, how="inner")
    df = df[[text_col, label_col]].dropna()
    
    print(f"Total labeled reviews: {len(df):,}")
    print("Class Distribution:")
    print(df[label_col].value_counts())

    print("\n[2/7] Preprocessing Text...")
    df[text_col] = df[text_col].apply(clean_text)
    df = df[df[text_col].str.len() > 10] # Filter very short/empty reviews
    
    X = df[text_col]
    y = df[label_col]

    print("\n[3/7] Train/Test Split (Stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("\n[4/7] Feature Engineering (TF-IDF N-grams)...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.9,
        sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("\n[5/7] Handling Class Imbalance (RandomOverSampler)...")
    ros = RandomOverSampler(random_state=42)
    X_train_res, y_train_res = ros.fit_resample(X_train_vec, y_train)
    print(f"Balanced training size: {X_train_res.shape[0]:,} samples")

    print("\n[6/7] Training & Hyperparameter Tuning (GridSearchCV)...")
    models = {
        "LogisticRegression": (LogisticRegression(max_iter=1000, class_weight="balanced"), {
            "C": [0.1, 1, 10]
        }),
        "LinearSVM": (LinearSVC(max_iter=2000, class_weight="balanced", dual=False), {
            "C": [0.1, 1, 10]
        })
    }

    best_f1 = 0
    best_model = None
    
    for name, (model, params) in models.items():
        print(f"  Tuning {name}...")
        grid = GridSearchCV(model, params, cv=3, scoring="f1_weighted", n_jobs=-1)
        grid.fit(X_train_res, y_train_res)
        
        y_pred = grid.predict(X_test_vec)
        f1 = f1_score(y_test, y_pred, average="weighted")
        acc = accuracy_score(y_test, y_pred)
        
        print(f"    Best Params: {grid.best_params_}")
        print(f"    F1 Score: {f1:.4f} | Accuracy: {acc:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_model = grid.best_estimator_

    print(f"\nFinal Model Selected: {best_model.__class__.__name__}")

    print("\n[7/7] Evaluation & Export...")
    y_final_pred = best_model.predict(X_test_vec)
    final_acc = accuracy_score(y_test, y_final_pred)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_final_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_final_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f"Improved Skin Classifier (Acc: {final_acc*100:.1f}%)")
    plt.savefig(os.path.join(LOG_DIR, "skin_model_v2_cm.png"))
    print(f"Confusion matrix saved to {LOG_DIR}/skin_model_v2_cm.png")

    # Save
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best_model, f)
    with open(VEC_PATH, "wb") as f:
        pickle.dump(vectorizer, f)
    
    print(f"Models saved: {MODEL_PATH}, {VEC_PATH}")

    # Final Comparison
    old_acc = 0.70  # From training_report.json
    improvement = ((final_acc - old_acc) / old_acc) * 100
    print("\n" + "="*40)
    print("  MODEL COMPARISON")
    print("="*40)
    print(f"  OLD Accuracy  : {old_acc*100:.1f}%")
    print(f"  NEW Accuracy  : {final_acc*100:.1f}%")
    print(f"  Improvement   : {improvement:+.2f}%")
    print("="*40)

if __name__ == "__main__":
    run_pipeline()
