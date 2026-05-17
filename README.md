🎓 Student Performance Prediction
A machine learning pipeline that predicts student academic outcomes (Pass/Fail) using demographic, behavioral, and socioeconomic features. The project covers the full ML workflow — from data generation and exploratory analysis to multi-model training, evaluation, and visualization.

📌 Table of Contents

Overview
Project Structure
Dataset
Features
Models
Bug Fixes
Results
Visualizations
Installation
Usage
Requirements
License


Overview
This project builds a binary classification system to predict whether a student will Pass or Fail based on 11 input features including study habits, attendance, parental education, and health status.
The pipeline includes:

Synthetic dataset generation (1,050 student records)
Exploratory Data Analysis (EDA)
Data cleaning and feature engineering
Training and comparing 3 ML models
Cross-validated evaluation with reproducible results
6 publication-quality visualizations


Project Structure
student_performance_prediction/
│
├── data/
│   ├── generate_dataset.py     # Generates synthetic student dataset
│   └── students.csv            # Generated dataset (1,050 records)
│
├── src/
│   └── train_model.py          # Full ML pipeline (EDA → Training → Evaluation)
│
├── outputs/                    # Auto-generated charts and plots
│   ├── 01_target_overview.png
│   ├── 02_feature_distributions.png
│   ├── 03_correlation_heatmap.png
│   ├── 04_model_comparison.png
│   ├── 05_feature_importance.png
│   └── 06_scatter_analysis.png
│
├── requirements.txt
└── README.md

Dataset
The dataset is synthetically generated using generate_dataset.py with numpy.random.seed(42) for reproducibility.
PropertyValueTotal Records1,050 studentsTarget Variablepass_fail (Pass / Fail)Missing Values~3–5% (intentional, realistic)Pass ThresholdFinal Score ≥ 50

Features
FeatureTypeDescriptionprev_scoreNumericalPrevious academic score (20–100)study_hours_per_dayNumericalAverage daily study hoursattendance_pctNumericalAttendance percentage (30–100%)sleep_hoursNumericalAverage nightly sleep hoursstudy_efficiencyEngineeredstudy_hours × attendance / 100parental_educationCategoricalNo Education → Postgraduate (0–4)health_statusCategoricalPoor → Excellent (0–3)family_incomeCategoricalLow / Medium / High (0–2)internet_accessBinaryYes = 1, No = 0extra_classesBinaryYes = 1, No = 0genderBinaryMale = 1, Female = 0

Models
Three classifiers are trained, cross-validated, and compared:
ModelDetailsLogistic RegressionPipeline with StandardScaler, max_iter=500Decision Treemax_depth=6, random_state=42Random Forest100 estimators, max_depth=10, n_jobs=-1
All models use 5-fold Stratified Cross-Validation with shuffle=True and random_state=42 for fully reproducible results.

Bug Fixes
This version (FIXED) resolves 7 critical bugs from the original implementation:
#BugFix1LabelEncoder inside df.assign() produced non-deterministic class mappingsReplaced with explicit dictionary: {'Fail': 0, 'Pass': 1}2cross_val_score for Logistic Regression used pre-scaled X_train, causing data leakageWrapped scaler + model in sklearn.Pipeline so scaling happens inside each CV fold3Hardcoded 'data/students.csv' path broke when run from any other directoryFixed using __file__-relative Path resolution4os.makedirs('outputs') created folder relative to CWDFixed to resolve path relative to script location5No validation for unknown categorical values — silently produced NaN featuresAdded explicit validation with descriptive ValueError before mapping6Model comparison bar chart y-axis limit was hardcoded to 108Fixed to be dynamic: max(all_accuracies) + 87cross_val_score had no random_state — results were non-reproducibleFixed with StratifiedKFold(shuffle=True, random_state=42)

Results

Exact accuracy values depend on the generated dataset. Below are representative results:

ModelTest AccuracyCV Accuracy (5-Fold)Logistic Regression~85%~84% ± 1%Decision Tree~87%~86% ± 1%Random Forest~91%~90% ± 1%
Random Forest achieves the best performance across both test and cross-validated accuracy.
Top Predictive Features (by importance):

Previous Score
Study Efficiency (engineered)
Attendance %
Study Hours/Day
Parental Education


Visualizations
All 6 charts are auto-saved to the outputs/ folder when the pipeline runs.
FileDescription01_target_overview.pngPass/Fail class distribution + Final Score histogram02_feature_distributions.pngDistribution of 4 numerical features after cleaning03_correlation_heatmap.pngLower-triangle correlation matrix of all features04_model_comparison.pngTest vs CV accuracy bar chart + Random Forest confusion matrix05_feature_importance.pngRandom Forest feature importance (horizontal bar chart)06_scatter_analysis.pngStudy Hours vs Prev Score and Attendance vs Final Score scatter plots

Installation
bash# 1. Clone the repository
git clone https://github.com/your-username/student-performance-prediction.git
cd student-performance-prediction

# 2. (Optional) Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

Usage
bash# Step 1: Generate the dataset (only needed once)
python data/generate_dataset.py

# Step 2: Run the full ML pipeline
python src/train_model.py
The pipeline will:

Load and explore the dataset
Clean missing values
Engineer and encode features
Split data (80% train / 20% test, stratified)
Train all 3 models
Evaluate and compare with cross-validation
Save 6 charts to outputs/
Print a final summary to console


Requirements
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
Python 3.8+ is required.

License
This project is licensed under the MIT License.
