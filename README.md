# SurvBoost: Two-Year Survival Prediction in Oral Cancer

This repository contains the implementation and analysis code for **SurvBoost**, a data-driven parametric survival gradient boosting framework for individualized two-year survival estimation in patients with oral cancer.

The associated study uses 10,500 oral cancer patient records from the Surveillance, Epidemiology, and End Results (SEER) database and evaluates survival prediction using the Concordance Index (C-Index) and Integrated Brier Score (IBS). The framework combines multiple parametric survival distributions with gradient-boosted decision trees and provides SHAP-based model interpretation.

> **Data availability:** The SEER patient-level dataset is not included in this repository. Researchers should obtain the required SEER data through the appropriate SEER access procedures and prepare the analysis dataset according to the study description.

## Key components

### SurvBoost model

SurvBoost learns patient-specific parameters of multiple parametric survival distributions using gradient-boosted decision trees. The final experimental configuration reported in the manuscript contains:

- **5 Weibull heads**
- **1 Log-Logistic head**
- **1 Log-Normal head**
- **5 Generalized Gamma heads**
- **250 boosting estimators**
- **Maximum tree depth:** 5
- **Learning rate:** 0.10
- **Elastic-Net regularization strength:** 0.01
- **L1/L2 mixing ratio:** 1
- **Head-weight activation:** ReLU
- **Parameter initialization:** random initialization with a controlled random state

The model learns patient-specific distribution parameters rather than directly predicting a scalar risk score. The distribution components are combined through a non-negative weighted composition of their hazards and cumulative hazards to obtain the individualized survival function.

## Survival distributions

The implementation includes the following parametric survival components:

- Weibull
- Log-Logistic
- Log-Normal
- Generalized Gamma

The Generalized Gamma component includes a custom differentiable implementation of the regularized upper incomplete gamma function to support gradient-based optimization of its distribution parameters.

## Explainable AI

The repository includes SHAP-based analysis for:

- Global feature importance
- Patient-level feature contributions
- Individual survival prediction interpretation

Kernel SHAP is used because its model-agnostic formulation is suitable for the proposed multi-head SurvBoost architecture.

## Repository contents

```text
SurvBoost/
├── README.md
├── model.py
└── SurvBoost.ipynb
```

- **`model.py`** — SurvBoost model implementation, parametric survival distributions, gradient-based parameter updates, and survival prediction functionality.
- **`SurvBoost.ipynb`** — analysis notebook covering model training/evaluation and SHAP-based interpretation.

## Target prediction

The primary prediction target is the individualized two-year survival probability:

\[
P(T > 24\text{ months} \mid X_i),
\]

where \(X_i\) denotes the clinical feature vector for patient \(i\).

## Clinical features

The study uses the following clinical features:

- Age
- Sex
- Race
- Marital status
- Grade
- T stage
- N stage
- M stage
- Radiation therapy
- Surgical resection

## Reproducibility

To reproduce the reported experiments:

1. Obtain the required SEER data through the appropriate SEER access procedures.
2. Prepare the analysis dataset according to the feature definitions and inclusion criteria described in the manuscript.
3. Place the prepared dataset in the location expected by the notebook.
4. Install the required Python dependencies for the model and analysis notebook.
5. Run `SurvBoost.ipynb` to train/evaluate the model and generate the SHAP analyses.

The patient-level SEER data are intentionally not distributed with this repository.

## Important note

This repository contains research code intended to support reproducibility of the associated study. The predictions produced by the model are intended for research and prognostic analysis and should not be interpreted as a substitute for clinical judgment or medical decision-making.

## Citation

If you use this implementation or build upon the SurvBoost framework, please cite the associated research article:

**Yadav, B. V., Mathew, R. T., Rajan, J., & V. A., J.** *Estimating Two-Year Survival Probability in Oral Cancer Using Clinical Features.*

The final bibliographic details and DOI will be added when the associated article is formally published.
