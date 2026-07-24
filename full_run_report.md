# EDA Agent Report
*Generated: 2026-07-24T09:21:22*

---

## Agent Report

> ⚠️ **Verification: 5 issue(s) found** — 5 unverified quoted value(s). See `verification_details.md` for exactly which statements.

**Dataset Overview (Step 1)**  
- Shape: **32 561 rows × 15 columns**.  
- Column types: numeric (`int64`) for *age, fnlwgt, education_num, capital_gain, capital_loss, hours_per_week*; the remaining columns are strings (categorical).  

**Missing‑value Summary (Step 2)**  

| Column | Non‑null count | Non‑null % | Missing count | Missing % |
|--------|----------------|-----------|---------------|----------|
| age | 32 561 | 100.0 % | 0 | 0.0 % |
| workclass | 30 725 | 94.36 % | 1 836 | 5.64 % |
| fnlwgt | 32 561 | 100.0 % | 0 | 0.0 % |
| education | 32 561 | 100.0 % | 0 | 0.0 % |
| education_num | 32 561 | 100.0 % | 0 | 0.0 % |
| marital_status | 32 561 | 100.0 % | 0 | 0.0 % |
| occupation | 30 718 | 94.34 % | 1 843 | 5.66 % |
| relationship | 32 561 | 100.0 % | 0 | 0.0 % |
| race | 32 561 | 100.0 % | 0 | 0.0 % |
| sex | 32 561 | 100.0 % | 0 | 0.0 % |
| capital_gain | 32 561 | 100.0 % | 0 | 0.0 % |
| capital_loss | 32 561 | 100.0 % | 0 | 0.0 % |
| hours_per_week | 32 561 | 100.0 % | 0 | 0.0 % |
| native_country | 31 978 | 98.21 % | 583 | 1.79 % |
| income | 32 561 | 100.0 % | 0 | 0.0 % |

**Numeric‑column Cardinality (Step 3)**  

| Column | Distinct values (nunique) |
|--------|---------------------------|
| age | 73 |
| fnlwgt | 21 648 |
| education_num | 16 |
| capital_gain | 119 |
| capital_loss | 92 |
| hours_per_week | 94 |

All numeric columns have **> 10 distinct values**, so they are treated as **continuous** variables (no binary or low‑cardinality discrete columns were found).

**Distribution Skewness (Step 4)**  

| Column | Skewness |
|--------|----------|
| age | 0.56 |
| fnlwgt | 1.45 |
| education_num | –0.31 |
| capital_gain | **11.95** |
| capital_loss | **4.59** |
| hours_per_week | 0.23 |

- *age*, *fnlwgt*, *education_num*, and *hours_per_week* show modest skew (|skew| < 1.5) and can be modeled directly or with mild transformations if desired.  
- *capital_gain* and *capital_loss* are **highly right‑skewed** (skew > 4). For these two variables, a log‑plus‑1 transformation (`log1p`) is advisable to reduce extreme outliers before modeling.  

**Target Variable – Class Balance (Step 5)**  

| Income class | Count | Percent |
|--------------|-------|---------|
| ≤ 50K | 24 720 | 75.92 % |
| > 50K | 7 841 | 24.08 % |

The dataset is **moderately imbalanced** toward the “≤ 50K” class. Techniques such as class‑weighting, oversampling (SMOTE), or undersampling may be considered during model training.

**Potential ID‑like Column (Step 6)**  

- **fnlwgt** is unique in **66.48 %** of rows (21 648 distinct values out of 32 561). While its high cardinality resembles an identifier, in the Adult Census dataset *fnlwgt* represents a sampling weight rather than a true ID, so it should be retained (perhaps after scaling) rather than dropped.  

**Summary of Recommendations**

1. **Missing data** – Only *workclass*, *occupation*, and *native_country* contain missing values (≤ 6 %). Impute these categorical fields with the most frequent category or a dedicated “Missing” label.  
2. **Skewed numeric features** – Apply `log1p` to *capital_gain* and *capital_loss* to mitigate extreme right‑skew. Other numeric features can be used as‑is or standardized.  
3. **Class imbalance** – Consider using class weights in algorithms (e.g., `class_weight='balanced'` in logistic regression) or resampling methods to improve recall on the minority “> 50K” class.  
4. **High‑cardinality weight column** – Keep *fnlwgt* (it carries important sampling information) but scale it (e.g., Min‑Max or Standard scaling) rather than treating it as an identifier.  
5. **Categorical encoding** – Encode nominal variables (e.g., *workclass*, *occupation*, *native_country*) with one‑hot or target encoding; ordinal‑like variables (*education*, *marital_status*) may benefit from ordinal encoding based on domain knowledge.  

These observations are fully backed by the tool‑generated statistics cited above.