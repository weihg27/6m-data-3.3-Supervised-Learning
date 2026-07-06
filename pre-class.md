# Before Class — L03 — Supervised Learning

**Estimated time: ~30 minutes.** Complete this before class.

This is the simplest version of "show up prepared": watch a short intro video, run one notebook, answer a few questions. You'll come to class having seen the idea in action.

| Step | Time | What you do |
|---|---|---|
| **1. Watch the intro video** | ~5 min  | [L03 intro video on YouTube](https://youtu.be/ZKjA1Xz8XOI) — sets up what supervised learning is and what you'll build in this lesson |
| **2. Try it** | ~20 min | Open and run `notebooks/01_monday_morning.ipynb` |
| **3. Reflect** | ~5 min  | Three short questions below |

---

## Step 0 — Watch the intro video (~5 min)

Before you open the notebook, watch the short intro video: **[L03 intro video on YouTube](https://youtu.be/ZKjA1Xz8XOI)**. It's under 5 minutes and frames the problem you're about to solve. Watching this first makes the notebook click faster.

🕹️ **After the video:** open the [interactive key-concepts page](https://su-ntu-ctp.github.io/6m-data-3.3-Supervised-Learning/) and play with it for 10–15 minutes. Drag the sliders, click the buttons — you can't break anything. Arriving in class having *seen* these ideas move makes the session far easier.

---

## Step 1 — Try it (~20 min)

Open **`notebooks/01_monday_morning.ipynb`** in VS Code with the `dsai-m3` kernel. Run every cell top to bottom. Read the markdown between cells. Don't skip any cell.

Sarah's job this week: predict which NorthStar customers will churn next quarter. The notebook hands her a 10,000-row customer dataset with 11 features and a `churned` label. She'll open the file the way a professional does — checking column types, missing values, and class balance — and discover that only ~12% of customers churned. That means a "model" that always predicts STAYED would score 88% accuracy while catching zero churners. *Accuracy is a lie when classes are imbalanced* — that realisation drives the whole week.

If this is your first time running a notebook in this repo, see [setup.md](./setup.md) once — you only need to do this for the first lesson.

---

## Step 2 — Quick reflection (~5 min)

Write a sentence or two for each. You can scribble in a notebook, in a journal, or just hold the answer in your head — what matters is that you *tried*.

**Q1. The 88% trap.**

Only ~12% of customers churned. Why would a "model" that always predicts STAYED score 88% accuracy — and why is that number worthless to Marcus?

**Q2. Threshold choice as a business decision.**

Sarah has a retention team that can call ~200 customers a week. If a model gives each customer a churn *probability*, how should that capacity number affect where Sarah draws the line between "call" and "don't call"? (One sentence — we build this properly in class.)

**Q3. The missing logins.**

About 8% of `last_login_days_ago` values are missing. Why might the *fact that a value is missing* itself be a useful clue about churn?

---

## Bring to class

- Your answer to Q1 in one sentence.
- One classification problem from your own domain where the positive class is rare (< 20%).

---

**Want to go deeper before class?** See **[reference.md](./reference.md) → *Further reading & watching*** at the bottom — videos and recommended readings that used to live in this guide.

**Ran out of time?** Doing just Step 1 (running the notebook) is enough. The class builds on having felt what the lesson teaches; the reflection questions get re-asked live.
