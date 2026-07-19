# Relax Inc. Take-Home Challenge — Predicting Future User Adoption

## Executive Summary

This project develops a machine learning model to predict whether a newly registered user will become an **adopted user**—defined as logging in on at least three separate days within any rolling seven-day period—using only information available at signup.

**Business Question**
> Which characteristics of newly registered users best predict future user adoption?

**Champion Model:** Tuned XGBoost

**Cross-Validation ROC-AUC:** **0.643 ± 0.013**

**Held-out Test ROC-AUC:** **0.649**

The results show that signup characteristics provide meaningful predictive power but explain only part of future adoption. This suggests that while acquisition channels matter, post-signup onboarding and engagement strategies are equally important for improving long-term user adoption.

---

# Data

The analysis uses two datasets covering **12,000 users** and **207,917 login events**.

| File | Description |
|------|-------------|
| `takehome_users.csv` | User attributes available at signup |
| `takehome_user_engagement.csv` | Historical login records used to construct the adoption label |

An adopted user is defined as someone who logs in on **three separate days within any rolling seven-day period**.

Overall, **1,602 of 12,000 users (13.4%)** met the adoption criterion. Because of this class imbalance, **ROC-AUC** was selected as the primary evaluation metric instead of accuracy.

---

# Methodology

This project follows the **CRISP-DM** framework.

### Data Preparation

To ensure a realistic production model, only information available **at the time of registration** was used.

The following variables were intentionally excluded:

- **last_session_creation_time** (target leakage)
- **account_age_days** (observation-window bias)
- **org_id** (high-cardinality feature that did not generalize well)

An ablation study comparing target encoding, LightGBM categorical handling, and feature exclusion confirmed that removing `org_id` produced the strongest generalization performance.

### Feature Engineering

Additional predictors included:

- Organization size
- Large organization indicator
- Invitation status
- Inviter activity
- Email domain
- Signup timing features

### Model Development

Four models were evaluated:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

Tree-based models were optimized using **Optuna** with five-fold cross-validation.

The champion model was selected exclusively from cross-validation performance and evaluated once on a held-out test set.

---

# Results

| Model | Mean CV ROC-AUC | Test ROC-AUC |
|------|------:|------:|
| **XGBoost (Champion)** | **0.643 ± 0.013** | **0.649** |
| Random Forest | 0.639 ± 0.013 | 0.644 |
| LightGBM | 0.633 ± 0.008 | 0.642 |
| Logistic Regression | 0.610 ± 0.013 | 0.620 |

The tuned ensemble models consistently outperformed the linear baseline, indicating that user adoption contains meaningful non-linear relationships.

---

# Key Findings

- **Signup source is the strongest actionable predictor** of future adoption.
- **Users from smaller organizations adopt more frequently** than those from larger organizations.
- **Signup month appeared highly predictive but primarily reflected observation-window bias rather than true business behavior.**
- **Marketing flags, signup weekday, and signup hour contributed little predictive value.**
- **The moderate predictive ceiling (ROC-AUC ≈ 0.65) suggests that signup information alone cannot fully explain adoption, highlighting the importance of post-signup engagement.**

Feature importance was validated using both gain-based importance and permutation importance on the held-out test set.

---

# Business Recommendations

- Prioritize onboarding for users predicted to have a low probability of adoption.
- Personalize onboarding based on signup source and organizational context.
- Invest in post-signup engagement initiatives since signup characteristics explain only part of user behavior.
- Redefine adoption using a fixed observation window in future analyses to eliminate remaining temporal bias.

---

# Repository Contents

| File | Description |
|------|-------------|
| `Relax Inc. Take-Home Challenge.ipynb` | Complete analysis, modeling, feature engineering, Optuna tuning, and interpretation |
| `Relax_Takehome_Writeup.md` | Executive summary of findings |
| `takehome_users.csv` | User dataset |
| `takehome_user_engagement.csv` | Login history dataset |

---

# How to Run

```bash
pip install pandas numpy scikit-learn xgboost lightgbm optuna category_encoders seaborn matplotlib shap

jupyter notebook "Relax Inc. Take-Home Challenge.ipynb"
```

The notebook is fully reproducible. Hyperparameter tuning requires approximately six minutes, while all remaining analyses execute within seconds.

---

# Author

**Michael Jumbo**

- Springboard Data Science Career Track
- Warwick MBA
- GitHub: https://github.com/MIJUMBO
