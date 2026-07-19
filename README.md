# Relax Inc. Take-Home Challenge — Predicting Future User Adoption

Predicting whether a newly registered user will become an **adopted user** — defined as logging in on three separate days within at least one rolling 7-day period — using only information available at signup.

**Business question:** Which characteristics of newly registered users best predict future user adoption?

## Data

Two tables covering 12,000 users who signed up over a two-year period:

| File | Contents |
|---|---|
| `takehome_users.csv` | User attributes: creation source, signup timestamp, email, organization, invitation, marketing flags |
| `takehome_user_engagement.csv` | 207,917 login records (user id, timestamp) |

The adoption label is constructed from the engagement table with a sliding 7-day window over each user's unique login days. **1,602 of 12,000 users (13.4%) qualify as adopted**, so accuracy is uninformative (~87% by predicting the majority class) and models are evaluated with ROC-AUC.

## Methodology

**Leakage discipline.** Only signup-time information enters the model:
- `last_session_creation_time` — excluded (post-signup information; target leakage)
- `account_age_days` — excluded (exposure confound: earlier signups had more time to meet the adoption criterion)
- `org_id` — excluded after a three-way **ablation** comparing cross-fitted target encoding, exclusion, and LightGBM native categorical handling; exclusion performed best, showing org-specific effects do not generalize beyond training organizations

**Feature engineering.** Invitation indicators, inviter activity (total invites sent by the user's inviter), organization size and large-org indicators, email domain (top domains + other), and signup timing.

**Modeling.** Logistic Regression baseline plus Random Forest, XGBoost, and LightGBM in a shared scikit-learn pipeline (one-hot encoding + standardization). Tree-model hyperparameters tuned with **Optuna** (TPE sampler, 5-fold CV on the training split only). Champion selected on **cross-validation** — never on the test set — with the held-out test score reported once as confirmation.

## Results

| Model | Mean CV ROC-AUC | Held-out Test ROC-AUC |
|---|---|---|
| **XGBoost (champion, tuned)** | **0.643 ± 0.013** | **0.649** |
| Random Forest (tuned) | 0.639 ± 0.013 | 0.644 |
| LightGBM (tuned) | 0.633 ± 0.008 | 0.642 |
| Logistic Regression | 0.610 ± 0.013 | 0.620 |

Tuned ensembles beat the linear baseline by ~2 standard deviations, indicating modest non-linear structure in the signal.

## Key findings

- **Signup pathway is the key actionable factor.** Adoption ranges from ~16.5% (guest invites) and ~15.1% (organization invites) down to ~7.9% (personal-project signups) — a more-than-twofold spread across creation sources.
- **Organizational context matters** — and users in *smaller* organizations adopt at higher rates than those in larger ones.
- **Signup month tops raw permutation importance but is substantially an observation-window artifact**, confirmed by a cohort diagnostic (adoption falls monotonically from 17.8% for the Oct 2013 cohort to 1.3% for May 2014, since recent signups had little time to qualify). It is retained as a control, not interpreted as actionable; a fixed-window adoption label is the recommended refinement.
- **Marketing flags, signup weekday, and signup hour contribute ~nothing** — the current email-marketing levers show no relationship with adoption.
- **The moderate ceiling (~0.65 AUC) is itself the insight:** signup-time attributes only partially determine adoption, so post-signup onboarding warrants at least equal investment with acquisition targeting.

Feature attribution uses **gain-based importance** (not split counts, which mechanically inflate continuous features) cross-checked with **permutation importance** on the held-out test set.

## Repository contents

| File | Description |
|---|---|
| `Relax Inc. Take-Home Challenge.ipynb` | Full analysis: label construction, EDA, feature engineering, ablation, Optuna tuning, model comparison, importance analysis |
| `Relax_Takehome_Writeup.md` | One-page summary of findings and recommendations |
| `takehome_users.csv`, `takehome_user_engagement.csv` | Input data (if not included, place them beside the notebook) |

## How to run

```bash
pip install pandas numpy scikit-learn xgboost lightgbm optuna category_encoders seaborn matplotlib shap
jupyter notebook "Relax Inc. Take-Home Challenge.ipynb"
```

Run all cells top to bottom. The Optuna tuning cell takes ~6 minutes (2-minute budget per tree model); all other cells run in seconds.

## Author

**Michael Jumbo** — [GitHub](https://github.com/MIJUMBO) · Warwick MBA · Springboard Data Science
