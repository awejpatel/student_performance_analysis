# 🎓 Student Performance Prediction
**By Awej Shaikh | BSc IT, BAMU University | 3rd Year**

---

## 📌 Project Overview

This project predicts whether a student will **Pass or Fail** based on academic and socio-economic features using three Machine Learning classification models. It follows a complete real-world data science pipeline from raw data to model evaluation.

**Tech Stack:** Python · pandas · NumPy · scikit-learn · Matplotlib · Seaborn

---

## 📁 Project Structure

```
student_performance_prediction/
│
├── data/
│   ├── generate_dataset.py   ← Generates synthetic student dataset (1050 records)
│   └── students.csv          ← Generated dataset
│
├── src/
│   └── train_model.py        ← Full ML pipeline (Steps 1–7)
│
├── outputs/                  ← All saved charts (auto-generated)
│   ├── 01_target_overview.png
│   ├── 02_feature_distributions.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_model_comparison.png
│   ├── 05_feature_importance.png
│   └── 06_scatter_analysis.png
│
└── README.md                 ← This file
```

---

## 📊 Dataset Details

| Property        | Value                      |
|-----------------|----------------------------|
| Total Records   | 1,050 students             |
| Total Features  | 13 columns                 |
| Target Variable | `pass_fail` (Pass / Fail)  |
| Missing Values  | ~3–5% in 4 columns (intentional, realistic) |

### Features in Dataset

| Column                | Type        | Description                          |
|-----------------------|-------------|--------------------------------------|
| `student_id`          | Integer     | Unique ID                            |
| `gender`              | Categorical | Male / Female                        |
| `parental_education`  | Ordinal     | No Education → Postgraduate (5 levels)|
| `family_income`       | Ordinal     | Low / Medium / High                  |
| `internet_access`     | Binary      | Yes / No                             |
| `extra_classes`       | Binary      | Yes / No (tuition / coaching)        |
| `health_status`       | Ordinal     | Poor / Average / Good / Excellent    |
| `prev_score`          | Float       | Previous exam score (20–100)         |
| `study_hours_per_day` | Float       | Daily study hours (0–12)             |
| `attendance_pct`      | Float       | Attendance percentage (30–100%)      |
| `sleep_hours`         | Float       | Hours of sleep per night (4–10)      |
| `final_score`         | Float       | Final exam score (target for reg.)   |
| `pass_fail`           | Binary      | Pass (≥50) / Fail (<50) ← **target** |

---

## 🔁 Pipeline — Step by Step

---

### ✅ STEP 1 — Load & Explore Data

**What we do:**
- Load the CSV with `pd.read_csv()`
- Check `df.shape`, `df.dtypes`, `df.head()`
- Count missing values with `df.isnull().sum()`
- Visualize the target class distribution and final score histogram

**Why it matters:**
Before doing anything, you must understand what data you have — how many rows, which columns have missing data, and whether the target classes are balanced or not.

**Key Finding:**
- 692 students Fail vs 358 Pass → **class imbalance** exists
- Missing values in: `prev_score` (27), `study_hours_per_day` (31), `attendance_pct` (43), `sleep_hours` (60)

📸 Output: `01_target_overview.png`

---

### ✅ STEP 2 — Data Cleaning (Missing Values)

**What we do:**
- Identify all columns with NaN using `df.isnull().sum()`
- Fill missing numerical values with the **median** of that column

```python
df[col] = df[col].fillna(df[col].median())
```

**Why median, not mean?**
Median is resistant to outliers. If a student's score is 20 or 95 (extreme values), the mean gets pulled toward them. Median stays at the middle regardless.

**Result:** 0 missing values after cleaning.

📸 Output: `02_feature_distributions.png`

---

### ✅ STEP 3 — Feature Engineering & Encoding

**What we do:**

**Ordinal Encoding** — for features with a natural order:
```python
parental_education: No Education=0, Primary=1, Secondary=2, Graduate=3, Postgraduate=4
health_status:      Poor=0, Average=1, Good=2, Excellent=3
family_income:      Low=0, Medium=1, High=2
```

**Binary Encoding** — for Yes/No columns:
```python
internet_enc  = 1 if internet_access == 'Yes' else 0
extra_class_enc = 1 if extra_classes == 'Yes' else 0
gender_enc    = 1 if gender == 'Male' else 0
```

**New Feature: Study Efficiency**
```python
study_efficiency = study_hours_per_day × attendance_pct / 100
```
This captures the combined impact of effort and presence — a student who studies 8 hours but attends 40% is less efficient than one who studies 5 hours and attends 90%.

**Why not One-Hot Encoding for ordinal features?**
One-Hot encoding treats all categories as equal (no order). Since "Postgraduate" is genuinely higher than "Primary", ordinal encoding preserves that meaning.

📸 Output: `03_correlation_heatmap.png`

---

### ✅ STEP 4 — Train / Test Split

**What we do:**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
```

| Split   | Samples |
|---------|---------|
| Train   | 840     |
| Test    | 210     |

**Why stratify=y?**
Without stratification, the test set might accidentally have 100% Fail cases. `stratify=y` ensures the same Pass/Fail ratio in both train and test sets.

**Why random_state=42?**
Ensures reproducibility. Every time the code runs, the same split is made.

**Scaling for Logistic Regression:**
```python
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)   # ← ONLY transform, never fit on test!
```
Logistic Regression is sensitive to feature scale. Without scaling, `prev_score` (20–100) would dominate over `gender_enc` (0 or 1).

---

### ✅ STEP 5 — Train 3 Models

```python
lr = LogisticRegression(max_iter=500, random_state=42)
dt = DecisionTreeClassifier(max_depth=6, random_state=42)
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
```

**Model 1 — Logistic Regression:**
A linear model that uses the sigmoid function to output a probability between 0 and 1. Predict Pass if probability ≥ 0.5. Simple and interpretable.

**Model 2 — Decision Tree:**
Splits data recursively based on feature thresholds (e.g., "prev_score > 55?"). Easy to visualize but prone to overfitting.

**Model 3 — Random Forest:**
An ensemble of 100 Decision Trees, each trained on a random subset of data (bagging) and random features. Final prediction = majority vote. Reduces variance and overfitting.

---

### ✅ STEP 6 — Evaluate & Compare Models

**Metrics used:**

| Metric    | Formula                             | What it tells you            |
|-----------|-------------------------------------|------------------------------|
| Accuracy  | Correct predictions / Total         | Overall correctness          |
| Precision | TP / (TP + FP)                      | Of predicted Pass, how many actually pass? |
| Recall    | TP / (TP + FN)                      | Of actual Pass students, how many did we catch? |
| F1-Score  | 2 × (Precision × Recall) / (P + R) | Balance between Precision & Recall |

**Results:**

| Model               | Test Accuracy | CV Accuracy (5-Fold) |
|---------------------|---------------|----------------------|
| Logistic Regression | 78.57%        | 80.83%               |
| Decision Tree       | 74.29%        | 76.55%               |
| **Random Forest**   | **80.00%**    | **78.69%**           |

**Why Random Forest wins:**
It uses 100 trees instead of 1. Each tree sees different rows and features. The noise in any single tree gets averaged out, leading to a more robust prediction.

📸 Output: `04_model_comparison.png`

---

### ✅ STEP 7 — Feature Importance (Random Forest)

Random Forest tells us how much each feature contributed to its decisions:

| Rank | Feature           | Importance |
|------|-------------------|------------|
| 1    | Previous Score    | 31.9%      |
| 2    | Attendance %      | 18.6%      |
| 3    | Study Efficiency  | 17.0%      |
| 4    | Study Hours/Day   | 9.9%       |
| 5    | Sleep Hours       | 8.3%       |
| 6    | Parental Education| 4.0%       |
| 7    | Health Status     | 3.2%       |
| 8    | Family Income     | 2.5%       |
| 9    | Extra Classes     | 1.7%       |
| 10   | Internet Access   | 1.5%       |
| 11   | Gender            | 1.3%       |

**Key Insight:** Academic history (`prev_score`) and consistent effort (`attendance`, `study efficiency`) are the strongest predictors. Demographic factors (gender, income) matter less.

📸 Output: `05_feature_importance.png`, `06_scatter_analysis.png`

---

## ▶️ How to Run

```bash
# Step 1: Install dependencies
pip install pandas numpy scikit-learn matplotlib seaborn

# Step 2: Generate dataset
cd student_performance_prediction
python data/generate_dataset.py

# Step 3: Run full ML pipeline
python src/train_model.py
```

All charts will be saved in the `outputs/` folder.

---

## 🧠 Key Learnings

1. **Data cleaning matters** — even 3–5% missing data can crash ML models if not handled
2. **Ordinal encoding** is better than one-hot for naturally ordered categories
3. **Feature engineering** (study_efficiency) improved model signal
4. **Random Forest outperforms** single models by reducing variance via ensemble learning
5. **CV accuracy** gives a more honest estimate than single train/test split alone

---

*Built by Awej Shaikh | shaikhawej10@gmail.com | github.com/awejshaikh*
