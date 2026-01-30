# Linear Regression (SLR & MLR)

This repo is a hands-on, easy-to-understand walkthrough of Simple & Multiple Regression using Python. If you're someone who loves clarity, clean explanations, and step-by-step reasoning (without drowning in jargon), this repo is for you.

Check out the complete article I have written on this Framework: [Regression Framework for Systolic Blood Pressure Prediction](https://nsdsda.medium.com/building-a-complete-regression-framework-for-systolic-blood-pressure-prediction-d454b813a0ad)

The goal is to understand regression deeply, by actually doing it, not just memorizing formulas.

We’ll explore:
* Simple Linear Regression
* Multiple Regression (2 predictors)
* Multiple Regression (4 predictors)
* Manual Beta Coefficient Computation (to see what’s exactly happening behind the scenes)
* Scikit-learn implementation
* SciPy & NumPy implementation

I'm gonna explain everything in a way you’d be able to understand each concepts intuitively, even if you're new to regression.

----

I told you not to worry about jargon. So here’s a quick glossary of terms we’ll use:
* **Regression:** A statistical method to model and analyze relationships between variables.
* **Simple Linear Regression (SLR):** Regression with one predictor variable.
* **Multiple Linear Regression (MLR):** Regression with two or more predictor variables.
* **Predictor/Independent Variable:** The variable(s) we use to predict the outcome.
* **Response/Dependent Variable:** The outcome variable we want to predict.
* **`β` (Beta Coefficient):** The slope(s) of the regression line(s), indicating how much the response variable changes with a one-unit change in the predictor.
* **Intercept (`β₀`):** The expected value of the response variable when all predictors are zero.
* **`R²` (R-squared):** A statistical measure that represents the proportion of the variance for the response variable explained by the predictor(s) in the model.


I also told not to memorize formulas. So, let me show you two important formulas we’ll use, 😂 just to get familiar.
1. **Simple Linear Regression Equation:** `Y = β₀ + β₁ × ε`
Where,
- Y: Response variable
- X: Predictor variable
- β₀: Intercept
- β₁: Slope (beta coefficient)
- ε: Error term (captures variability not explained by the model)

2. **Multiple Linear Regression Equation:** `Y = β₀ + β₁ × X₁ + β₂ × X₂ + ... + βₙ × Xₙ + ε`
Where,
- `Y`: Response variable
- `X₁, X₂, ..., Xₙ`: Predictor variables
- `β₀`: Intercept
- `β₁, β₂, ..., βₙ`: Slope coefficients for each predictor
- `ε`: Error term
- `n`: Number of predictors

`β` coefficients indicate the change in `Y` for a one-unit change in the corresponding `X`, holding other predictors constant.

----

## 📂 Repository Structure

```
📁 regression-project/
│
├── 01_slr.ipynb
├── 02_mlr_2_predictors.ipynb
├── 03_mlr_4_predictors.ipynb
├── 04_manual_beta_computation.ipynb
├── 05_sklearn_lr.ipynb
├── 06_scipy_numpy_lr.ipynb
└── README.md
```

Each notebook builds naturally on the previous one, so you can follow along step-by-step.

----

## 1. Simple Linear Regression

**Goal:** Understand the core idea of regression: How one variable affects another?

We cover:
* What regression actually is?
* How `β₀` and `β₁` (intercept & slope) are computed by the model?
* How to interpret coefficients in real life?
* Visualizing regression lines
* Evaluating model fit (`R²`)

Example used:
**Age → Systolic Blood Pressure (SBP)**
Because in real life, SBP usually increases with age, a relatable pattern.


## 2. Multiple Regression (2 Predictors)

Now we add more factors, because real-world outcomes rarely depend on just one thing.

Example used:
**Age + BMI → SBP**

Concepts covered:
* How adding predictors changes interpretation?
* How multiple `β`s work together?
* How to check which predictor matters more?
* Difference between single vs. multiple regression intuition
* Model summary interpretation (statsmodels)

I have kept explanations approachable in this notebook.


## 3. Multiple Regression (4 Predictors)

Here we go one level deeper. More predictors = more realism.

Example used:
**Age + BMI + ActivityLevel + SaltIntake → SBP**

This notebook helps you understand:
* How regression handles many variables?
* Why coefficients change when new predictors enter?
* Feature relationships and multicollinearity (lightly explained)
* Overfitting risk with too few samples
* How to keep models meaningful as they grow

Still, I have tried to make it friendly, clear, and practical enough.😉


## 4. Manual Beta Computation

Here, I have explained about manual computation of `β` to understand:
* How regression actually finds the best line?
* What least squares means?
* Why the model chooses those specific coefficients?
* How predictions are formed from coefficients?

Everything is step-by-step, with explanations at each stage.
* Compute means
* Compute covariance
* Compute variance
* Derive `β₁` → then `β₀`
* Validate with the library output

This is actually great for strengthening intuition.


## 5. Scikit-learn Linear Regression

Finally, we implement regression using Scikit-learn, the most popular ML library in Python.

We keep things simple and focus on:
- fitting the model with just a few lines of code
- checking coefficients and intercept
- making predictions
- comparing results with our previous models

This notebook shows the same concept from a more practical ML angle, giving our project multiple approaches instead of just one.


## 6. SciPy x NumPy Linear Regression

In this notebook we implement regression using SciPy and NumPy, focusing on the mathematical underpinnings. While earlier notebooks focused on StatsModels and Scikit-learn, here we explore the fast, scientific-computing style approach.

What’s inside?
* Simple Linear Regression using `scipy.stats.linregress`
* Multiple Linear Regression using `np.linalg.lstsq`
* Manual β-coefficient extraction using the least-squares equation

    $$
    \beta = (X^\top X)^{-1} X^\top y
    $$

* Lightweight prediction workflow
* Comparison with StatsModels and Scikit-learn

This notebook is perfect for those who want to see regression from a more mathematical and computational perspective, while still being easy to follow. Again, I have prioritized clarity and intuition over complexity.

----

Explore the detailed case studies that document the analysis, methodology, and outcomes of this project:

- Case Study 1: [00_LR.md](https://github.com/nibeditans/Data-Projects/blob/main/Case%20Studies/Project-Based/00_LR.md)
- Case Study 2: [01_LR.md](https://github.com/nibeditans/Data-Projects/blob/main/Case%20Studies/Project-Based/01_LR.md)
- Case Study 3: [02_LR.md](https://github.com/nibeditans/Data-Projects/blob/main/Case%20Studies/Project-Based/02_LR.md)


## What to Learn from this Project?
* How regression models find relationships?
* How to interpret coefficients logically?
* How multiple predictors interact?
* When models overfit?
* Why sample size matters?
* How data shape affects results?
* How to compute `β`s manually (the raw Math intuition)?

I have tried to explain in plain language, with clarity prioritized over complexity.

## Tech Stack

* **Python 3**
* **NumPy**
* **Pandas**
* **Statsmodels**
* **Matplotlib/Seaborn** for visualization
* **Scikit-learn**
* **SciPy**

Everything is written in Jupyter Notebooks for interactivity and explanation-friendly formatting. Okay?😉

This project isn’t a huge end-to-end ML pipeline, but it definitely covers some of the most important concepts in Data Science with depth, simplicity, and intuition.

Feel free to fork, clone, modify, or build on top of it. Learning grows best through experimentation. 🌱

For more interesting Projects, you can check out my complete [Data Science & Analytics Projects Collection](https://github.com/nibeditans/Data-Projects).

----
