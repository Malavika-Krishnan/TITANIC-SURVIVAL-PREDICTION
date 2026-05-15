# 🚢 Titanic Survival Prediction

A machine learning system for predicting passenger survival on the Titanic using classical ML models, structured feature engineering, and comprehensive model evaluation.

---

## Overview

The Titanic disaster remains one of history's most studied events in data science. This project explores how passenger attributes — such as age, gender, class, and fare — can be used to predict survival outcomes using classical machine learning.

The repository demonstrates a complete machine learning workflow:

- Exploratory Data Analysis
- Data Preprocessing
- Feature Engineering
- Model Training
- Hyperparameter Optimization
- Cross-Validation
- Performance Evaluation

---

## Architecture

### Dataset Features

| Feature | Description |
|---|---|
| `Pclass` | Passenger ticket class (1st, 2nd, 3rd) |
| `Sex` | Gender of the passenger |
| `Age` | Age in years |
| `SibSp` | Number of siblings / spouses aboard |
| `Parch` | Number of parents / children aboard |
| `Fare` | Ticket fare paid |
| `Embarked` | Port of embarkation (C, Q, S) |
| `Cabin` | Cabin number (heavily missing) |
| `Survived` | Target label — 0 = No, 1 = Yes |

### Models Used

| Model | Purpose |
|---|---|
| Linear Regression | Regression baseline |
| Ridge Regression | Regularized regression baseline |
| XGBoost Classifier | Main predictive model |
| Random Forest Classifier | Ensemble predictive model |
| GridSearchCV | Hyperparameter optimization |
| K-Fold Cross Validation | Model validation |

### Tech Stack

`Python` · `Scikit-learn` · `XGBoost` · `Pandas` · `NumPy` · `Matplotlib` · `Seaborn` · `Plotly` · `Jupyter Notebook`

---

## Repository Structure

```
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Titanic — Repository Structure</title>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Syne:wght@400;600;700&display=swap" rel="stylesheet"/>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg:        #0d0f14;
    --surface:   #13161e;
    --border:    #1e2330;
    --purple-50: #EEEDFE;
    --purple-400:#7F77DD;
    --purple-600:#534AB7;
    --teal-50:   #E1F5EE;
    --teal-400:  #1D9E75;
    --teal-600:  #0F6E56;
    --amber-50:  #FAEEDA;
    --amber-400: #BA7517;
    --amber-600: #854F0B;
    --coral-50:  #FAECE7;
    --coral-400: #D85A30;
    --coral-600: #993C1D;
    --text:      #e8e6de;
    --muted:     #7a7870;
    --dim:       #3a3830;
  }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'JetBrains Mono', monospace;
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 48px 24px;
  }

  .card {
    width: 100%;
    max-width: 680px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    overflow: hidden;
  }

  .card-header {
    padding: 20px 28px;
    border-bottom: 1px solid var(--border);
    display: flex;
    align-items: center;
    gap: 12px;
  }

  .dots { display: flex; gap: 7px; }
  .dot {
    width: 12px; height: 12px;
    border-radius: 50%;
  }
  .dot-red    { background: #ff5f57; }
  .dot-yellow { background: #febc2e; }
  .dot-green  { background: #28c840; }

  .card-title {
    font-family: 'Syne', sans-serif;
    font-size: 13px;
    font-weight: 600;
    color: var(--muted);
    letter-spacing: 0.04em;
    margin-left: 4px;
  }

  .tree { padding: 32px 28px 36px; }

  /* Root row */
  .root-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 24px;
  }
  .root-label {
    font-family: 'Syne', sans-serif;
    font-size: 15px;
    font-weight: 700;
    color: var(--purple-400);
    letter-spacing: 0.01em;
  }
  .root-badge {
    font-size: 10px;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 99px;
    background: #1c1a2e;
    color: var(--purple-400);
    border: 1px solid #2e2a52;
    letter-spacing: 0.05em;
    text-transform: uppercase;
  }

  /* File group */
  .file-group {
    position: relative;
    display: flex;
    gap: 0;
    margin-bottom: 6px;
  }

  /* Vertical trunk for the whole group */
  .trunk-wrap {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 24px;
    flex-shrink: 0;
  }
  .trunk-top   { width: 1px; flex: 0 0 22px; background: var(--dim); }
  .trunk-mid   { width: 1px; flex: 1; background: var(--dim); }
  .trunk-none  { width: 1px; flex: 1; }

  /* Horizontal branch to file pill */
  .branch-wrap {
    display: flex;
    flex-direction: column;
    flex: 1;
    padding-bottom: 16px;
  }

  .branch-row {
    display: flex;
    align-items: center;
    gap: 0;
  }

  .branch-line {
    width: 20px;
    height: 1px;
    background: var(--dim);
    flex-shrink: 0;
    margin-top: 1px;
  }

  /* File pill */
  .file-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 8px 16px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
    border: 1px solid;
    cursor: default;
    transition: opacity 0.15s;
    user-select: none;
    white-space: nowrap;
  }
  .file-pill:hover { opacity: 0.8; }

  .pill-teal  { background: #0d1f1a; border-color: var(--teal-600);  color: var(--teal-400); }
  .pill-amber { background: #1c1509; border-color: var(--amber-600); color: var(--amber-400); }
  .pill-coral { background: #1e0f09; border-color: var(--coral-600); color: var(--coral-400); }

  .pill-icon { font-size: 14px; }

  /* Sub-items under a file */
  .sub-items {
    margin-top: 8px;
    padding-left: 20px; /* aligns with pill left edge */
    display: flex;
    flex-direction: column;
    gap: 0;
  }

  .sub-item {
    display: flex;
    align-items: flex-start;
    gap: 0;
    position: relative;
  }

  /* Sub vertical trunk */
  .sub-trunk {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 18px;
    flex-shrink: 0;
    align-self: stretch;
  }
  .sub-trunk-line { width: 1px; flex: 1; background: var(--dim); }
  .sub-trunk-last { width: 1px; flex: 0 0 14px; background: var(--dim); }

  /* Sub horizontal branch */
  .sub-branch {
    width: 14px;
    height: 1px;
    background: var(--dim);
    flex-shrink: 0;
    margin-top: 13px;
  }

  .sub-text {
    font-size: 12px;
    color: var(--muted);
    padding: 5px 0 5px 6px;
    line-height: 1.4;
  }

  .sub-text .tag {
    display: inline-block;
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 1px 6px;
    border-radius: 4px;
    margin-right: 4px;
    vertical-align: middle;
  }
  .tag-teal  { background: #0d1f1a; color: var(--teal-400);  border: 1px solid #0f3828; }
  .tag-amber { background: #1c1509; color: var(--amber-400); border: 1px solid #2e2008; }
  .tag-coral { background: #1e0f09; color: var(--coral-400); border: 1px solid #2e1208; }
</style>
</head>
<body>

<div class="card">
  <div class="card-header">
    <div class="dots">
      <div class="dot dot-red"></div>
      <div class="dot dot-yellow"></div>
      <div class="dot dot-green"></div>
    </div>
    <span class="card-title">titanic-survival-prediction / repository structure</span>
  </div>

  <div class="tree">

    <!-- Root -->
    <div class="root-row">
      <span style="color:var(--muted);font-size:13px;">📁</span>
      <span class="root-label">titanic-survival-prediction/</span>
      <span class="root-badge">root</span>
    </div>

    <!-- train.csv -->
    <div class="file-group">
      <div class="trunk-wrap">
        <div class="trunk-top"></div>
        <div class="trunk-mid"></div>
      </div>
      <div class="branch-wrap">
        <div class="branch-row">
          <div class="branch-line"></div>
          <div class="file-pill pill-teal">
            <span class="pill-icon">📊</span> train.csv
          </div>
        </div>
        <div class="sub-items">
          <!-- child 1 -->
          <div class="sub-item">
            <div class="sub-trunk"><div class="sub-trunk-line"></div></div>
            <div class="sub-branch"></div>
            <div class="sub-text"><span class="tag tag-teal">rows</span> 891</div>
          </div>
          <!-- child 2 -->
          <div class="sub-item">
            <div class="sub-trunk"><div class="sub-trunk-line"></div></div>
            <div class="sub-branch"></div>
            <div class="sub-text"><span class="tag tag-teal">cols</span> 12 features</div>
          </div>
          <!-- child 3 (last) -->
          <div class="sub-item">
            <div class="sub-trunk"><div class="sub-trunk-last"></div></div>
            <div class="sub-branch"></div>
            <div class="sub-text"><span class="tag tag-teal">label</span> Survived column included</div>
          </div>
        </div>
      </div>
    </div>

    <!-- test.csv -->
    <div class="file-group">
      <div class="trunk-wrap">
        <div class="trunk-top"></div>
        <div class="trunk-mid"></div>
      </div>
      <div class="branch-wrap">
        <div class="branch-row">
          <div class="branch-line"></div>
          <div class="file-pill pill-teal">
            <span class="pill-icon">📊</span> test.csv
          </div>
        </div>
        <div class="sub-items">
          <div class="sub-item">
            <div class="sub-trunk"><div class="sub-trunk-line"></div></div>
            <div class="sub-branch"></div>
            <div class="sub-text"><span class="tag tag-teal">rows</span> 418</div>
          </div>
          <div class="sub-item">
            <div class="sub-trunk"><div class="sub-trunk-line"></div></div>
            <div class="sub-branch"></div>
            <div class="sub-text"><span class="tag tag-teal">cols</span> 11 features</div>
          </div>
          <div class="sub-item">
            <div class="sub-trunk"><div class="sub-trunk-last"></div></div>
            <div class="sub-branch"></div>
            <div class="sub-text"><span class="tag tag-teal">label</span> unlabeled — for prediction</div>
          </div>
        </div>
      </div>
    </div>

    <!-- gender_submission.csv -->
    <div class="file-group">
      <div class="trunk-wrap">
        <div class="trunk-top"></div>
        <div class="trunk-mid"></div>
      </div>
      <div class="branch-wrap">
        <div class="branch-row">
          <div class="branch-line"></div>
          <div class="file-pill pill-amber">
            <span class="pill-icon">📄</span> gender_submission.csv
          </div>
        </div>
        <div class="sub-items">
          <div class="sub-item">
            <div class="sub-trunk"><div class="sub-trunk-last"></div></div>
            <div class="sub-branch"></div>
            <div class="sub-text"><span class="tag tag-amber">type</span> sample submission file</div>
          </div>
        </div>
      </div>
    </div>

    <!-- notebook -->
    <div class="file-group" style="margin-bottom:0;">
      <div class="trunk-wrap">
        <div class="trunk-top"></div>
        <div class="trunk-none"></div>
      </div>
      <div class="branch-wrap" style="padding-bottom:0;">
        <div class="branch-row">
          <div class="branch-line"></div>
          <div class="file-pill pill-coral">
            <span class="pill-icon">📓</span> titanic-survival-prediction.ipynb
          </div>
        </div>
        <div class="sub-items">
          <div class="sub-item">
            <div class="sub-trunk"><div class="sub-trunk-line"></div></div>
            <div class="sub-branch"></div>
            <div class="sub-text"><span class="tag tag-coral">phase 1</span> EDA · preprocessing · feature engineering</div>
          </div>
          <div class="sub-item">
            <div class="sub-trunk"><div class="sub-trunk-last"></div></div>
            <div class="sub-branch"></div>
            <div class="sub-text"><span class="tag tag-coral">phase 2</span> model training · evaluation · results</div>
          </div>
        </div>
      </div>
    </div>

  </div>
</div>

</body>
</html>
```

---

## Installation

**Clone the repository**
```bash
git clone https://github.com/your-username/titanic-survival-prediction.git
cd titanic-survival-prediction
```

**Install dependencies**
```bash
pip install pandas numpy scipy matplotlib seaborn plotly scikit-learn xgboost jupyter
```

**Run the notebook**
```bash
jupyter notebook titanic-survival-prediction.ipynb
```

---

## Key Features

- Binary survival classification from passenger metadata
- Age group binning and feature engineering
- Correlation heatmaps and survival distribution visualizations
- Feature importance analysis via Random Forest
- Cross-validation evaluation
- Hyperparameter optimization with GridSearchCV

---

## Future Scope

- Integration with larger historical passenger datasets
- Deep learning-based classification (MLP, TabNet)
- Interactive survival prediction dashboard
- Advanced ensemble and stacking methods
- SHAP-based model interpretability

---

## Data Source

Dataset from the [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic).

---

## License

This project is open-source and available under the [MIT License](LICENSE).
