# 🚢 Titanic Survival Prediction

A machine learning project that predicts passenger survival on the Titanic using exploratory data analysis, feature engineering, and multiple classification models.

---

## 📂 Project Structure

```
titanic-survival-prediction/
│
├── 📊 train.csv                          →  891 rows · 12 features · labeled
├── 📊 test.csv                           →  418 rows · 11 features · unlabeled
├── 📄 gender_submission.csv              →  sample submission file
└── 📓 titanic-survival-prediction.ipynb  →  EDA · preprocessing · modeling

---

## 📊 Dataset Overview

| Dataset | Rows | Columns |
|---------|------|---------|
| train.csv | 891 | 12 |
| test.csv | 418 | 11 |

**Features:**

| Column | Description |
|--------|-------------|
| `PassengerId` | Unique passenger identifier |
| `Survived` | Target — 0 = No, 1 = Yes |
| `Pclass` | Ticket class (1st, 2nd, 3rd) |
| `Name` | Passenger name |
| `Sex` | Gender |
| `Age` | Age in years |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Ticket` | Ticket number |
| `Fare` | Passenger fare |
| `Cabin` | Cabin number |
| `Embarked` | Port of embarkation (C, Q, S) |

**Missing Values (Training Set):**
- `Cabin` — 687 missing (77%)
- `Age` — 177 missing (20%)
- `Embarked` — 2 missing

---

## 🔍 Exploratory Data Analysis

- **Survival rate:** 38.4% survived (342 out of 891)
- **Gender:** Females had a 74.2% survival rate vs. 18.9% for males
- **Passenger class:** Higher-class passengers had significantly better survival odds
- **Age groups:** Survival rates analyzed across age bins (Infants → Old)
- **Correlation heatmap** used to visualize relationships between numeric features

---

## ⚙️ Data Preprocessing

- Dropped `Cabin` column (too many missing values)
- Imputed missing `Age` values using iterative/KNN imputation
- Filled missing `Embarked` with mode
- Encoded categorical features (`Sex`, `Embarked`, `AgeGroup`) using `LabelEncoder`
- Created age group buckets: Infants, Toddlers, Kids, Teens, Youngs, Middle Aged, Old

---

## 🤖 Models Used

### 1. Linear Regression (baseline)
- Features: `Pclass`, `Age`, `Fare`
- Used as a regression baseline

### 2. Ridge Regression
- Features: `Pclass`, `Age`, `Fare`
- Evaluated using RMSE

### 3. XGBoost Classifier
- Wrapped in a `Pipeline` with `MinMaxScaler`
- Hyperparameter tuning via `GridSearchCV` (5-fold CV)
- Parameters tuned: `n_estimators`, `learning_rate`

### 4. Random Forest Classifier
- Wrapped in a `Pipeline` with `MinMaxScaler`
- Hyperparameter tuning via `GridSearchCV` (5-fold CV)
- Parameters tuned: `n_estimators`, `max_depth`

**Evaluation Metrics:** Accuracy, F1 Score, Precision, Confusion Matrix

---

## 🛠️ Libraries & Tools

| Library | Purpose |
|---------|---------|
| `pandas`, `numpy` | Data manipulation and computation |
| `scipy` | Statistical analysis |
| `matplotlib`, `seaborn`, `plotly` | Data visualization |
| `scikit-learn` | Preprocessing, modeling, evaluation |
| `xgboost` | Gradient boosting classifier |

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/titanic-survival-prediction.git
cd titanic-survival-prediction
```

### 2. Install dependencies
```bash
pip install pandas numpy scipy matplotlib seaborn plotly scikit-learn xgboost
```

### 3. Launch the notebook
```bash
jupyter notebook titanic-survival-prediction.ipynb
```

---

## 📁 Data Source

The dataset is from the [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic).

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
