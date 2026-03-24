# Task 1 — Data Exploration and Preprocessing

---

## Overview

This project presents a complete **Exploratory Data Analysis (EDA) and Preprocessing pipeline** applied to a restaurant dataset. The work covers data structure inspection, quality assessment, type conversions, missing value treatment, and a thorough analysis of the target variable — `Aggregate rating` — including class imbalance detection.

This forms the foundational step of the data science workflow, ensuring the dataset is clean, well-understood, and ready for feature engineering and model development in subsequent tasks.

---

## Project Structure

```
task1_project/
│
├── Dataset_.csv                                  ← Raw input dataset
├── Task1_Data_Exploration_Preprocessing.ipynb    ← Main Jupyter notebook
│
└── outputs/                                      ← Generated visualizations
    ├── fig1_target_distribution.png
    ├── fig2_missing_values.png
    ├── fig3_class_imbalance.png
    ├── fig4_boxplot_numerics.png
    ├── fig5_correlation_heatmap.png
    └── fig6_rating_by_country.png
```

---

## Dataset at a Glance

| Property              | Value                          |
|-----------------------|--------------------------------|
| Total Records         | 9,551 rows                     |
| Total Features        | 21 columns                     |
| Numerical Features    | 11                             |
| Categorical Features  | 10                             |
| Missing Values        | 9 (in `Cuisines` column only)  |
| Target Variable       | `Aggregate rating` (0.0 – 4.9)|

---

## Technologies Used

| Tool / Library | Purpose                                      |
|----------------|----------------------------------------------|
| Python 3.10    | Core programming language                    |
| Pandas         | Data loading, manipulation, and analysis     |
| NumPy          | Numerical operations and array handling      |
| Matplotlib     | Chart and figure generation                  |
| Seaborn        | Statistical visualizations and heatmaps      |
| Jupyter        | Interactive notebook environment             |

---

## Task Breakdown

### Step 1 — Dataset Exploration
- Loaded the CSV dataset using Pandas
- Identified dataset dimensions: **9,551 rows × 21 columns**
- Printed all column names, data types, and a 5-row preview
- Generated descriptive statistics for all numerical features

### Step 2 — Missing Value Analysis and Handling

| Column    | Missing Count | Missing % | Strategy Applied              |
|-----------|---------------|-----------|-------------------------------|
| Cuisines  | 9             | 0.09%     | Filled with `'Unknown'`       |
| All others| 0             | 0.00%     | No action required            |

Dropping rows was avoided to preserve all 9,551 records. The `'Unknown'` placeholder clearly signals absent cuisine data for downstream processing.

### Step 3 — Data Type Conversion

Four columns stored binary information as `Yes` / `No` strings. These were converted to integer format (`1` / `0`) to ensure compatibility with numerical operations and machine learning models.

| Column               | Before   | After  |
|----------------------|----------|--------|
| Has Table booking    | object   | int64  |
| Has Online delivery  | object   | int64  |
| Is delivering now    | object   | int64  |
| Switch to order menu | object   | int64  |

### Step 4 — Target Variable Analysis

The target variable `Aggregate rating` ranges from **0.0 to 4.9**.

> **Important:** A rating of `0.0` does **not** indicate a poor restaurant — it means the restaurant has not yet received any user rating. These 2,148 entries (22.5%) are labelled `"Not rated"` and must be treated separately in modelling.

| Metric            | Value   |
|-------------------|---------|
| Minimum           | 0.0     |
| Maximum           | 4.9     |
| Mean (rated only) | 3.19    |
| Median            | 3.20    |
| Std Deviation     | 1.52    |

### Step 5 — Class Imbalance Detection

| Category  | Count | Share   |
|-----------|-------|---------|
| Average   | 3,737 | 39.12%  |
| Not rated | 2,148 | 22.49%  |
| Good      | 2,100 | 21.99%  |
| Very Good | 1,079 | 11.30%  |
| Excellent |   301 |  3.15%  |
| Poor      |   186 |  1.95%  |

**Imbalance ratio: ~20:1** (largest class vs. smallest class)

The dataset exhibits significant class imbalance. If left unaddressed, any classifier trained on this data will heavily favour the majority class, leading to misleading accuracy scores and poor performance on minority classes.

**Recommended mitigation strategies:**
- Stratified train-test splitting to preserve class ratios
- SMOTE (Synthetic Minority Oversampling Technique) for minority class augmentation
- Class weights (`class_weight='balanced'`) during model training
- Evaluation using F1-score, Precision-Recall AUC, and Cohen's Kappa rather than raw accuracy

---

## Visualizations

| File                          | Description                                           |
|-------------------------------|-------------------------------------------------------|
| `fig1_target_distribution.png`| Rating category bar chart + score histogram           |
| `fig2_missing_values.png`     | Column-wise missing value count (post-handling)       |
| `fig3_class_imbalance.png`    | Pie chart of rating category proportions              |
| `fig4_boxplot_numerics.png`   | Box plots for key numerical features (outlier view)   |
| `fig5_correlation_heatmap.png`| Pearson correlation matrix heatmap                    |
| `fig6_rating_by_country.png`  | Average rating by country code (top 15)               |

---

## How to Run

1. **Install dependencies**
   ```bash
   pip install pandas numpy matplotlib seaborn jupyter
   ```

2. **Launch the notebook**
   ```bash
   jupyter notebook Task1_Data_Exploration_Preprocessing.ipynb
   ```

3. **Run all cells**
   ```
   Kernel → Restart & Run All
   ```

   All output figures are saved automatically to the `outputs/` folder.

> The notebook also ships with **pre-rendered outputs** embedded in every cell, so it can be viewed fully without running a single line of code — in Jupyter, VS Code, or Google Colab.

---

## Key Findings

| Finding | Detail |
|---------|--------|
| Dataset is largely complete | Only 9 missing values across 9,551 records |
| Binary encoding applied | 4 columns standardised to 0/1 integer format |
| Target variable has a special zero class | 22.5% of records are unrated (0.0) — treat separately |
| Votes correlates most with rating | Pearson r = +0.31 — higher-voted restaurants tend to score higher |
| Significant class imbalance | 20:1 ratio between largest and smallest rating class |

---

## Next Steps

- **Task 2:** Feature engineering — encode categorical variables, scale numerical features, handle the unrated class
- **Task 3:** Model building — apply classification or regression models with appropriate class-balancing strategies
- **Task 4:** Model evaluation — assess performance using imbalance-aware metrics

---

*Data Science Internship — Task 1 Submission*
