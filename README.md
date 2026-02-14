# Assignment2_ML

# Bank Marketing Subscription Prediction 
# PROBLEM STATEMENT  
Predict whether a client will subscribe to a term deposit (`y` ∈ {no, yes}) using the UCI *Bank Marketing* dataset. The task is binary classification.

**Dataset description**  
- Source: UCI Machine Learning Repository — *Bank Marketing*.  
- Size: ≥ 41,188 instances, ≥ 20 input features (depending on file used).  
- Target: `y` (yes/no) — imbalanced (≈ 11% positive).  
- Files typically used: `bank-additional-full.csv` (41,188 rows, 20 inputs + target) or `bank-full.csv` (45,211 rows, 16 inputs + target).

## Models used
- Logistic Regression
- Decision-tree Classifier
- K-Nearest Neighbor Classifier
- Naive-Bayes Classifier
- Random-forest(Ensemble)
- XGBoost(Ensemble)

## Comparison Table for Evaluation Metrics:

| Model                         | Accuracy |   AUC    | Precision | Recall  |   F1     |   MCC     |
|------------------------------|---------:|---------:|----------:|--------:|---------:|----------:|
| **Logistic Regression**      | 0.831836 | 0.802949 | 0.362027  | 0.646552| 0.464157 | 0.395839  |
| **Decision Tree Classifier** | 0.845675 | 0.618833 | 0.318790  | 0.325431| 0.322076 | 0.235029  |
| **K-Nearest Neighbor Classifier** | 0.899086 | 0.771649 | 0.636535  | 0.242816| 0.351534 | 0.351119  |


| Model | Observation |
|---|---|
| **Logistic Regression** | Strong **recall** (0.647) and good **AUC** (0.803) indicate reliable ranking and positive-class capture; **precision** is moderate (0.362), so expect some false positives. Solid baseline; threshold tuning could lift precision without losing too much recall. |
| **Decision Tree Classifier** | Lowest overall balance: **AUC** (0.619), **precision** (0.319), **recall** (0.325), and **F1** (0.322) are all weak—suggests under/overfitting with current settings; prefer tuning or using an ensemble. |
| **K‑Nearest Neighbor Classifier** | **Highest accuracy** (0.899) and **precision** (0.637) but **low recall** (0.243) → very conservative (few false positives, many misses). Good when false positives are costly; try different *k* or distance weighting to boost recall. |
| **Naive Bayes Classifier (Gaussian)** | Emphasizes positives: **recall** (0.629) is high with fair **AUC** (0.776), but **precision** is low (0.319) and accuracy is the lowest (0.807). Useful when missing positives is more harmful than extra false alarms. |
| **Ensemble Model – Random Forest** | **Best overall**: top **AUC** (0.814), **F1** (0.507), and **MCC** (0.442) with high **recall** (0.630). Most balanced choice out of the box; good default, with room to tune for higher precision. |
| **Ensemble Model – XGBoost** | Good balance across metrics—**AUC** (0.792), **recall** (0.624), **F1** (0.464)—slightly behind Random Forest. Likely to improve with tuning (depth, learning rate, number of trees). |
