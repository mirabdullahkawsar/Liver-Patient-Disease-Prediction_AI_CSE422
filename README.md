# Liver Patient Disease Prediction

A **machine learning classification project** built with **Python, Pandas, and Scikit-learn** that predicts whether a patient has liver disease based on clinical and demographic data from the **Liver Patient Dataset (LPD)**.

## 📊 Dataset

The project uses the **Liver Patient Dataset (LPD)**, which contains patient records with features such as:

* Age and Gender of the patient
* Total / Direct Bilirubin
* Alkaline Phosphotase (Alkphos)
* Alamine Aminotransferase (Sgpt)
* Aspartate Aminotransferase (Sgot)
* Total Proteins
* Albumin (ALB)
* Albumin and Globulin Ratio (A/G Ratio)
* `Result` — the target label indicating presence of liver disease

> Place `Liver Patient Dataset (LPD)_train.csv` in the project directory (or update the file path in the notebook) before running.

## ✨ What This Project Does

### 1. Data Cleaning & Preprocessing
* Loads the dataset and inspects shape, nulls, and data types
* Drops columns with more than 50% missing values
* Fills missing values:
  * Numerical/quantitative features → mean imputation
  * Categorical features → mode imputation
* Cleans column names (strips whitespace, replaces spaces with underscores)
* Checks for negative values in quantitative columns

### 2. Exploratory Data Analysis (EDA)
* Histograms of each feature, split by target class (`Result`)
* Overall feature distribution histograms
* Gender distribution pie chart
* Feature-vs-target bar plots
* Correlation heatmaps (before and after preprocessing)
* Before/after comparison of missing values

### 3. Feature Engineering
* One-hot encoding of the categorical `Gender_of_the_patient` column
* Feature scaling with `StandardScaler` (including a batch/partial-fit example)
* Train/test split (70/30 and 80/20 variants shown)

### 4. Model Training & Evaluation
Multiple models are trained and compared on the task:

| Model | Type |
| ----- | ---- |
| K-Nearest Neighbors | Classification |
| Decision Tree | Classification |
| Random Forest | Classification |
| Logistic Regression | Classification |
| Naive Bayes (GaussianNB) | Classification |
| Linear Regression | Regression (rounded to nearest class for comparison) |

Each model is evaluated using:
* **Accuracy score**
* **Classification report** (precision, recall, F1-score)
* **Confusion matrix** (visualized as a heatmap)

## 🏆 Achievements / Results

Actual results from the notebook run (test set: 6,139 samples):

| Model | Accuracy |
| ----- | -------- |
| 🥇 **Random Forest** | **99.77%** |
| 🥈 Decision Tree | 99.15% (99.09% in a second run) |
| 🥉 K-Nearest Neighbors | 91.61% (92% in classification report) |
| Logistic Regression | 73% |
| Linear Regression (rounded to nearest class) | 71.4% (72% in classification report) |
| Naive Bayes (GaussianNB) | 56% |

**Best model: Random Forest — 99.77% accuracy**, with Decision Tree close behind at ~99%. Both tree-based models massively outperform the linear/probabilistic models on this dataset.

Per-class performance (label `1` = disease, `2` = no disease) from the classification reports:

* **K-Nearest Neighbors** — precision/recall around 0.94–0.95 for class 1 and 0.84–0.86 for class 2; weighted F1 ≈ 0.92
* **Decision Tree** — precision/recall ≈ 0.99 for class 1 and ≈ 0.98–0.99 for class 2; weighted F1 ≈ 0.99
* **Logistic Regression** — strong on class 1 (recall 0.94) but weak on class 2 (recall only 0.19), showing it struggles to catch minority-class disease cases
* **Linear Regression** — essentially predicts class 1 for almost everyone (class 2 recall ≈ 0.00), which inflates accuracy without being a useful classifier
* **Naive Bayes** — the opposite pattern: strong recall on class 2 (0.96) but weak on class 1 (0.40), giving the lowest overall accuracy

Additional analysis performed:

* 📈 Correlation heatmaps generated **before and after** preprocessing, to show how cleaning, encoding, and scaling affect feature relationships
* 🔍 A before/after comparison table of missing values across all columns, confirming the imputation step worked as intended
* 🌳 Decision tree structure exported with `graphviz` for visualization
* 🧩 Confusion matrices plotted as heatmaps for KNN, Decision Tree, Logistic Regression, Linear Regression, and Naive Bayes

## 🛠️ Technologies

* **Python**
* **Pandas** / **NumPy**
* **Matplotlib** / **Seaborn**
* **Scikit-learn**
  * `SimpleImputer`, `StandardScaler`, `RobustScaler`, `LabelEncoder`
  * `train_test_split`
  * `KNeighborsClassifier`, `DecisionTreeClassifier`, `RandomForestClassifier`
  * `LogisticRegression`, `LinearRegression`, `GaussianNB`
  * `accuracy_score`, `classification_report`, `confusion_matrix`, `mean_squared_error`, `r2_score`
* **Graphviz** (decision tree export)

## 📂 Project Structure

```text
CSE422-Project/
│
├── 422_notun_project.ipynb   # Main notebook: preprocessing, EDA, model training & evaluation
├── Liver Patient Dataset (LPD)_train.csv   # Dataset (not included — add your own copy)
└── README.md
```

## 🚀 Run the Project

Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn graphviz
```

Then open and run the notebook (originally developed in Google Colab):

```bash
jupyter notebook 422_notun_project.ipynb
```

If running outside Colab, update the dataset path in the notebook from `/content/Liver Patient Dataset (LPD)_train.csv` to your local file path.

## 🎓 Project Focus

This project was built for a **Computer Intelligence / Machine Learning (CSE422)** course and demonstrates:

* Real-world data cleaning and missing-value handling
* Exploratory data analysis and visualization
* Categorical encoding and feature scaling
* Train/test splitting for model evaluation
* Comparing multiple classification algorithms on the same task
* Model evaluation using accuracy, classification reports, and confusion matrices

## 👨‍💻 Project

**CSE422 — Liver Patient Disease Prediction**
**Language:** Python
**Environment:** Google Colab / Jupyter Notebook# CSE422-Project
