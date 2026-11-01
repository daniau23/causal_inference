# Causal Inference: S-Learners, T-Learners & Cross-Fitting

In the absence of running an A/B test experiment, this small, hands-on repo shows how to use ordinary machine learning models to estimate a causal effect from observational data.

> **Does offering a $20 discount actually stop customers from churning?**

Instead of just predicting churn, we estimate the **causal effect** of the discount by comparing what would happen if everyone got it versus if no one did. This is useful when an experiment is impractical, too slow, or simply was never run, and you have to work with data that already exists. The repo walks through two classic meta-learners, S-learner and T-learner, side by side on the same synthetic customer dataset, then shows how to make each one more reliable with **cross-fitting**.

![S-learner vs T-learner](images/s_vs_t_learner.png)

---

## Why this repo exists

A normal ML model learns *correlations*. But the retention team gives discounts mostly to high-risk customers, so simply comparing churn rates between discounted and non-discounted customers is misleading. That's **selection bias**, caused by **confounders** — things like tenure, app logins, support tickets, and spend that influence both who gets a discount and who churns.

The data in this scenario is **historical** and **observational**. Historical describes *when* it was collected; observational describes *how* the discount was assigned — by the retention team, not at random. That distinction matters: in an A/B test, random assignment makes the groups comparable, while with observational data we have to model our way around the imbalance.

![Confounding](images/confounding.png)

Causal inference corrects for this by asking a counterfactual question for every customer:

- What is the churn risk if this customer **gets** the discount?
- What is the churn risk if this customer **does not** get the discount?

The average difference between those two predictions is the **Average Treatment Effect (ATE)**.

---

## What's inside

| Notebook | Approach | What it does |
| --- | --- | --- |
| [`s_learner.ipynb`](s_learner.ipynb) | **S-learner** | Trains **one** Random Forest with the treatment as an extra feature, then predicts twice. |
| [`t_learner.ipynb`](t_learner.ipynb) | **T-learner** | Trains **two** separate Random Forests, one for the control group and one for the treated group. |
| [`cross_fitting_s_learner.ipynb`](cross_fitting_s_learner.ipynb) | **Cross-fitted S-learner** | Runs 5-fold cross-fitting with one treatment-aware model per fold, then refits on the full training set. |
| [`cross_fitting_t_learner.ipynb`](cross_fitting_t_learner.ipynb) | **Cross-fitted T-learner** | Runs 5-fold cross-fitting with separate treated/control models inside each fold, then refits both models. |

The two standard notebooks follow the same structure:

1. Generate 10,000 synthetic subscription customers with realistic confounders.
2. Split the data 80/20 into train and test sets.
3. Tune a Random Forest with 5-fold cross-validation and grid search.
4. Create two counterfactual worlds: everyone treated, and everyone in control.
5. Average the difference in predicted churn to get the ATE.

The cross-fitting notebooks keep the same data generator and train/test split, but change the estimation step:

1. Split the **training set** into 5 stratified folds.
2. Train on four folds and predict counterfactuals on the held-out fold.
3. For the S-learner, fit one model per fold with `discount_given` as a feature. For the T-learner, split each training fold by treatment status and fit two models.
4. Collect the out-of-fold individual effects and average them into a **cross-fitted ATE**.
5. Refit the model(s) on the entire training set, evaluate on the untouched test set, and compute the final test-set ATE.

![Workflow](images/workflow.png)

---

## The two approaches at a glance

**S-learner (single model)** — put the treatment column in with the other features and train one model. To estimate the effect, ask the model to predict the same customers twice: once with the discount switched on, once switched off. Simple and fast, but tree-based models can sometimes underuse the treatment column if other features are very predictive.

**T-learner (two models)** — split the training data by treatment status and fit a separate model to each group. Because the treatment is never a feature, the models are forced to learn the two groups independently. This avoids the S-learner's "ignored treatment" risk, at the cost of training two models and having less data for each.

---

## Why cross-fitting?

Fitting a model and estimating each customer's effect on the same data can be too optimistic: the model has already seen that customer's outcome. Cross-fitting reduces this leakage. Each customer's counterfactual prediction comes from a model trained without them, and the held-out fold rotates so every customer gets an out-of-fold estimate.

![Cross-fitting](images/cross_fitting.png)

In this repo, cross-fitting is implemented with `StratifiedKFold(n_splits=5, shuffle=True, random_state=50)`. For the cross-fitted S-learner, the notebook also reports out-of-fold predictive metrics so you can compare the validation and test performance directly. Cross-fitting does **not** remove confounding by itself — it improves how the models are evaluated. The causal assumptions below still apply.

---

## Before you trust the estimate

Without randomisation, a causal estimate is only as good as its assumptions. The main ones are:

- **No unmeasured confounding** — every variable that affects both the discount and churn is measured and included.
- **Overlap (positivity)** — every kind of customer has a realistic chance of appearing in both the treated and control groups.
- **Consistency** — "receiving the discount" means the same thing for every customer.

> Machine learning does not create causality by itself. It helps estimate the counterfactual outcomes once the study design and these assumptions are reasonable.

---

## Results

All four methods land in the same neighbourhood, which is a good sign that the effect is stable:

![ATE comparison](images/ate_comparison.png)

| Model | Cross-fitted ATE (out-of-fold) | Final test ATE | Plain-English meaning |
| --- | --- | --- | --- |
| S-learner | — | **-5.8 percentage points** | The discount lowers churn probability by about 5.8 points. |
| T-learner | — | **-6.9 percentage points** | The discount lowers churn probability by about 6.9 points. |
| Cross-fitted S-learner | **-6.0 percentage points** | **-5.8 percentage points** | Cross-fitting reproduces the S-learner estimate. |
| Cross-fitted T-learner | **-6.5 percentage points** | **-6.9 percentage points** | Cross-fitting reproduces the T-learner estimate. |

![Cross-fitted ATE](images/cross_fitting_ate.png)

The cross-fitted S-learner also reports out-of-fold predictive quality:
| Metrics| Cross-fitted ATE (out-of-fold) | Test |
| --- | --- | --- | 
| ROC-AUC |0.70 | 0.71 |
| Precision| 0.66| 0.67 |
| Recall | 0.71 | 0.72 |
| F1 | 0.68 | 0.69 |

Because churn is coded as `1 = churned`, a **negative** effect is a good thing: the discount pushes churn down. A customer with a 20% baseline risk drops to roughly 14.2% under the S-learner estimate.

![Churn risk example](images/churn_risk_example.png)

---

## Quick start

The notebooks are self-contained and use only common Python data-science libraries.

```bash
pip install numpy pandas scikit-learn matplotlib
```

Then open any notebook and run the cells top to bottom. The standard `s_learner.ipynb` and `t_learner.ipynb` notebooks use grid search, so they take a few minutes on a typical laptop. The cross-fitting notebooks use fixed Random Forest settings (`max_depth=10`, 150–250 trees) instead of grid search and run faster.

---

## Repo structure

```
causal_inference/
├── README.md
├── s_learner.ipynb
├── t_learner.ipynb
├── cross_fitting_s_learner.ipynb
├── cross_fitting_t_learner.ipynb
└── images/
    ├── s_vs_t_learner.png
    ├── workflow.png
    ├── confounding.png
    ├── cross_fitting.png
    ├── ate_comparison.png
    ├── cross_fitting_ate.png
    └── churn_risk_example.png
```
---

## A note on the data

The dataset is **synthetic** and generated inside the notebooks, so results are reproducible with a fixed seed. The data generator plants a known causal effect, which makes it a safe sandbox for checking whether each learner recovers the right direction and rough magnitude. Real-world observational work would also need careful study design, overlap checks, and sensitivity analysis for unmeasured confounding.