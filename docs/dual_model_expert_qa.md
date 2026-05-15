# Expert Interview Guide: Dual-Model ML Architecture

This guide prepares you for high-level technical questions regarding the **Biagiotti Dual-Model System**, specifically the transition from the V1 baseline to the V2 optimized skin classifier.

---

### 🧠 Core Architecture Questions

#### 1. "Why did you implement a dual-model system instead of just replacing the old model?"
**Expert Answer:**
"Replacing a production model blindly is risky. We adopted a **Blue-Green deployment pattern** at the code level. The V1 model serves as a stable, high-availability baseline. The V2 model provides superior performance but is gated by a **Confidence Threshold**. This ensures that the system always prioritizes reliability over novelty. If the 'Advanced' model isn't confident, the system remains 'Safe' by yielding to the baseline."

#### 2. "Why was Softmax needed for the LinearSVC model's output?"
**Expert Answer:**
"LinearSVC is a Support Vector Machine implementation that optimizes for the maximal margin. Unlike Probabilistic Classifiers (like Logistic Regression or Random Forest), it returns raw **Decision Scores** (the distance from the hyperplane), not probabilities. To make these scores interpretable for the UI and the fallback logic, I implemented a **Softmax Normalization** layer. This squashes the raw scores into a valid probability distribution (summing to 1.0), allowing us to set an intuitive confidence threshold."

#### 3. "What was your justification for the 0.70 confidence threshold?"
**Expert Answer:**
"The 0.70 threshold was empirically derived. I ran a validation pass on a sample of 500 real reviews and analyzed the score distribution. We found that 0.70 represents a 'High-Certainty' filter. While it triggers a fallback in ambiguous cases (~70% of the time on mixed text), it ensures that when the V2 model *does* take control, its predictions are mathematically robust. It serves as a **Quality Gate** to prevent 'hallucinations' in the classifier."

---

### 🛠️ Optimization & Training Questions

#### 4. "How did you address class imbalance in the V2 training set?"
**Expert Answer:**
"Our raw dataset was heavily skewed toward 'Normal' skin types. To prevent the model from becoming biased toward the majority class, I applied **Random Over-Sampling (ROS)** to the training split. By virtually balancing the counts of the 'Oily' and 'Sensitive' classes before fitting the SVM, we improved the **Recall** for minority skin types without sacrificing overall precision."

#### 5. "What was the significance of moving from Unigrams to Bigrams in the TF-IDF vectorizer?"
**Expert Answer:**
"In cosmetic reviews, context depends on qualifiers. A unigram approach sees 'oily' and 'not' separately. A **Bigram (N-gram 1,2)** approach captures phrases like 'less oily' or 'very dry'. This structural change was a primary driver in moving the model's accuracy from the 70% baseline to the 89%+ range."

---

### 🚀 Production & Performance Questions

#### 6. "Does running two models increase the latency of the API?"
**Expert Answer:**
"Minimal. Both models and their vectorizers are loaded as **Singletons** into memory at startup. In the default mode (V1), there is zero overhead. In 'Advanced' mode, even with the fallback check, the combined inference time remains under 15ms because the TF-IDF sparse matrix operations and SVM linear kernels are extremely computationally efficient."

#### 7. "If you were to further improve the system, what would be the next step?"
**Expert Answer:**
"I would implement **Continuous Active Learning**. By logging instances where the V2 model fell below the 0.70 threshold, we can identify exactly which types of reviews the system finds 'difficult.' We could then target those specific data points for manual labeling and use them to 'fine-tune' the V3 model."

---

> [!TIP]
> **Key keywords to use during your interview:**
> - Softmax Normalization
> - Confidence-based Fallback
> - Decision Function
> - Linear Kernel Efficiency
> - Balanced F1-Score
