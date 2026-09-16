Oral cancer is a major public health challenge worldwide, contributing signifi-
cantly to morbidity, mortality, and socioeconomic burden. Accurate estimation
of survival probability is essential for patient risk stratification, treatment plan-
ning, and clinical decision-making. This study proposes SurvBoost, a data-driven
survival gradient boosting model for estimation of two-year survival probabil-
ity using clinical features. The study utilizes 10,500 oral cancer patient records
from the Surveillance, Epidemiology, and End Results (SEER) database. A data-
driven distribution selection technique is employed, based on goodness-of-fit
analysis, which compares the theoretical cumulative hazard functions of candi-
date survival distributions with the empirical Nelson–Aalen cumulative hazard.
SurvBoost learns patient-specific distribution parameters using gradient-boosted
decision trees by optimizing right-censored survival likelihood with Elastic-Net
regularization. The model is evaluated using the Concordance Index (C-Index)
and Integrated Brier Score (IBS), achieving a C-Index of 0.7466 and an IBS of
0.1629, thereby outperforming the existing survival models. To enhance clinical
interpretability, SHapley Additive exPlanations (SHAP) are employed to provide
global feature importance and patient-specific explanations of survival predic-
tions. The proposed model offers an interpretable and data-driven approach for
two-year survival estimation in patients with oral cancer.

# SurvBoost + SHAP: Two-Year Survival Prediction

This notebook implements:

1. **SurvBoost** with data-driven distribution selection (Weibull + LogNormal + Generalized Gamma+ Log-logistic)
2. **Final model training** and test set evaluation
3. **SHAP (SHapley Additive exPlanations)** — model-agnostic XAI method:
   - **Overall Feature Importance** bar plot (summary plot)
   - **Patient-specific SHAP** bar plot showing positive/negative feature contributions
4. **New patient prediction** of P(T > 24 months | patient features)

---

**Dataset:** NCI SEER Oral Cancer Dataset  
**Model:** SurvBoost (Weibull + LogNormal + Generalized Gamma heads + Log-Logistic)  
**Target:** P(T > 24 months | patient features)
