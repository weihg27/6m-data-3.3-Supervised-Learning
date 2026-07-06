# L03 — Supervised Learning Foundations

> *Sarah Chen's third week at NorthStar Retail. Friday last week, Marcus asked: "Can you build us a model that predicts churn from our OWN customer data?" This week she trains her first model.*
> By the end of this lesson you will be able to take a messy real-world dataset, prepare it for modelling, train a classifier, evaluate it with the right metrics, and make a business-aware threshold decision.

---

## Before you start — environment setup

> **If this is your first time with this course, do this before anything else.**
>
> Follow the [**Setup Guide →**](./setup.md) to install the Python environment. It takes 10–15 minutes. The same environment works for all 10 lessons.
>
> **Already set up from L01/L02?** Your `dsai-m3` environment should already have everything (`scikit-learn` is included).

---

## Where L03 fits

| Lesson | What you build | What you carry forward |
|---|---|---|
| **L01 — Intro to ML** | Run a pre-trained sentiment model on 10,000 reviews | A working model someone else trained — and Marcus's question: *"Can you train your own?"* |
| **L02 — Probability & Statistics** | Bracket every number with confidence intervals + A/B test the apology coupon | The statistical lens to judge a model's output honestly |
| **L03 — Supervised Learning Foundations** *(you are here)* | Train your first model from scratch — predict churn on NorthStar's own customer data | A reproducible preprocessing → training → evaluation pipeline you can apply to any tabular ML problem |
| **L04 — Advanced Supervised Learning** | Decision trees, ensembles, Kaggle-style hyperparameter tuning | The toolkit most production tabular-ML systems use |

**The narrative thread:** Marcus's question — "*train YOUR OWN model on NorthStar data*" — is finally answered this week.

---

## What you will be able to do by the end

- **Prepare** a real-world tabular dataset for modelling: handle missing values, scale numeric features, encode categorical features, build a single sklearn `Pipeline`
- **Train** a logistic-regression classifier with a proper train/test split, and use k-fold cross-validation to estimate its performance more reliably than from one split alone
- **Evaluate** a classifier with the confusion matrix and the four metrics it generates: accuracy, precision, recall, F1
- **Choose** a classification threshold that matches the business objective — not just accept the 0.5 default
- **Explain** — to a manager — what your model gets right, what it gets wrong, and what a change in threshold would cost or save

---

## How class time is structured (~3 hrs)

| Phase | Time | Format |
|---|---|---|
| Concept recap (slides) | ~45–60 min | Instructor recaps the key concepts with the lesson slides — you already explored the [interactive key-concepts page](https://su-ntu-ctp.github.io/6m-data-3.3-Supervised-Learning/) pre-class |
| Hands-on code-alongs | ~90 min | Three notebooks (~25–30 min each) — Core sections only |
| Class exit survey | ~15 min | Quick survey to capture what clicked and what didn't — helps shape the next session |
| (Self-study after class) | self-paced | Each notebook has a 🟡 Extension section for going deeper |

**Why this structure?** Realistic 3-hour pacing means ~1 hour of slide-based concept recap + ~1.5 hours of coding *including Q&A and environment troubleshooting*, closing with a 15-minute exit survey. Each in-class notebook ends at a clearly marked 🟡 Extension boundary — anything below the line is for self-study, not class time.

---

## Your learning path

This lesson follows a three-phase flow. Work through the phases in order.

---

### Phase 1 — Before class: self-study (~30 min)

**Goal:** *Open the dataset*. See the mess. Form a question.

**Start here →** [**pre-class.md**](./pre-class.md)

You will:
- Open and run `notebooks/01_monday_morning.ipynb` (~15 min) — Sarah opens the new NorthStar churn dataset and sees the messiness of real data for the first time
- Reflect on which features you'd actually trust
- Watch two short videos (StatQuest on logistic regression + bias-variance)
- Explore the [**interactive key-concepts page**](https://su-ntu-ctp.github.io/6m-data-3.3-Supervised-Learning/) after the videos (GitHub Pages)
- Try three mini-exercises with sample answers

---

### Phase 2 — In class: concept review + hands-on notebooks (~3 hrs)

**Goal:** Build the pipeline end-to-end with the instructor — preprocess, train, evaluate, decide.

**Short reference & review →** [**lesson.md**](./lesson.md) (overview, key takeaways, threshold-choice checklist, 9-question review, L04→L10 course map)

**Notebooks — run in order:**

| # | Notebook | Sarah's day | What you explore |
|---|---|---|---|
| 02 | [`02_preprocessing.ipynb`](./notebooks/02_preprocessing.ipynb) | Tuesday | Missing values · scaling · encoding · sklearn Pipeline + ColumnTransformer |
| 03 | [`03_train_validate.ipynb`](./notebooks/03_train_validate.ipynb) | Wednesday | Train/test split · logistic regression · k-fold cross-validation · coefficient interpretation |
| 04 | [`04_metrics_threshold.ipynb`](./notebooks/04_metrics_threshold.ipynb) | Thursday | Confusion matrix · precision/recall/F1 · why accuracy alone is misleading · choosing the operating threshold |

Each notebook opens with a business scenario, guides you through the code with **Pause & Predict** prompts, and ends with a summary table and "Up next" pointer. Read every markdown cell, not just the code.

---

### Phase 3 — After class: assignment + further reading (self-paced)

**Goal:** Transfer what you learned to a completely new domain.

Sarah is loaned out to **Lakeside Bank**, where **Tom Bradley** (Head of Analytics) wants a churn model for credit-card customers. Same pipeline — different data, different feature meanings, different business cost of a false negative.

**Assignment →** [`notebooks/assignment.ipynb`](./notebooks/assignment.ipynb)

Three tiers of practice (guided → partial → open) followed by three independent exercises in a hospital readmission scenario. Sample solutions are at the bottom — attempt each exercise yourself before checking them.

**Further reading →** [**reference.md**](./reference.md)

---

## Core vs Optional — what this lesson teaches

**🟢 Core (taught in class):**
- Preprocessing pipeline — scaling, encoding, imputation, ColumnTransformer
- Train/test split + k-fold cross-validation
- Confusion matrix + precision/recall/F1 + threshold choice

**🟡 Optional (self-study, not assessed):**
- Bias-variance math + learning curves
- ROC curves + AUC derivation
- Manual feature engineering
- Coefficient confidence intervals

Optional material lives in [`notebooks/optional_extensions.ipynb`](./notebooks/optional_extensions.ipynb). Skipping it will not affect your understanding of later lessons.

---

## File map

```
README.md                              ← You are here
setup.md                               ← One-time environment setup
pre-class.md                           ← Phase 1: 30-min self-study guide
lesson.md                              ← Short reference: overview, takeaways, threshold checklist, review Q&A, course map
reference.md                           ← Phase 3: Further reading + glossary (~20 terms)
environment.yml                        ← Conda environment spec
docs/
  index.html                           ← Interactive key-concepts page — explore during pre-class (served at https://su-ntu-ctp.github.io/6m-data-3.3-Supervised-Learning/ via GitHub Pages)
notebooks/
  data/
    northstar_churn.csv                ← 10,000-row training dataset (loads into all 4 notebooks)
  01_monday_morning.ipynb              ← Pre-class hook: Sarah opens the data (~15 min, before class)
  02_preprocessing.ipynb               ← Part 1: Preprocessing (Tuesday, in class)
  03_train_validate.ipynb              ← Part 2: Train + Cross-validate (Wednesday, in class)
  04_metrics_threshold.ipynb           ← Part 3: Metrics + Threshold (Thursday, in class)
  assignment.ipynb                     ← After class: Lakeside Bank + hospital exercises
  optional_extensions.ipynb            ← 🟡 Optional: bias-variance · ROC-AUC · learning curves · feature engineering
```
