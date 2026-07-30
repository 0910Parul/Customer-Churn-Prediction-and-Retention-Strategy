# Predicting Customer Churn to Drive Retention Strategy

This project predicts which bank customers are **dissatisfied and at risk of leaving**, so the bank can intervene before they churn. Using the Santander customer dataset (anonymized behavioral features), it builds a machine-learning model to **rank customers by churn risk** and frames the results as a targeted, testable retention strategy.

## 🔍 Project Goals

- Predict which customers are at risk of leaving (dissatisfied), early enough to act.
- Handle severe class imbalance and choose the right success metric.
- Segment customers by risk and translate the model into a targeted, testable retention plan.

## 📄 Files Included

- **`Churn_Prediction_Presentation.pdf`** — Final presentation deck.
- **`Churn_Prediction_Presentation.mp4`** — Recorded presentation walkthrough.
- **`Top100_Features_correlation.ipynb`** — feature analysis and correlation study.
- **`data_dictionary.md`** — feature groups / prefix taxonomy for the anonymized columns.
- **`rf_submission_final.csv`** — Random Forest scored submission.
- **`requirements.txt`** — Python dependencies.

> Data is public via Kaggle and **not stored in this repo** — download from the [competition page](https://www.kaggle.com/competitions/santander-customer-satisfaction/data).

## 🧪 Methods Used

- Data cleaning (removing constant and duplicate features)
- Feature engineering and one-hot encoding
- Random Forest classification
- Class-imbalance handling
- Model evaluation with AUC-ROC

## 📈 Key Findings

- 96% of customers are satisfied, so **accuracy is misleading** — a model that predicts "satisfied" for everyone scores 96% but catches **zero** at-risk customers; we evaluated with **AUC-ROC** instead.
- The tuned **Random Forest** reliably separates at-risk from satisfied customers (cross-validated AUC ≈ 0.84; final leaderboard **AUC 0.827**) — **ranked #1 in class**.
- Applied to the bank's ~160M customers, **~6.4M are at risk**; retaining even 10% protects an estimated **$256M in annual revenue**.

## 🛠 Technologies

- Python (pandas, scikit-learn)
- Random Forest, cross-validation
- Jupyter Notebook
- AUC-ROC evaluation

## ✅ Outcome

The bank can flag dissatisfied customers early and target retention offers — with a **90-day A/B testing roadmap** to validate which interventions actually save customers.

---

By **Parul Chaudhary** · [LinkedIn](https://www.linkedin.com/in/parulchaud) · [Email](mailto:parul.jaswant@gmail.com)
