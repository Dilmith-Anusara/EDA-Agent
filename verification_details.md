# Verification Details (audit trail)
*Generated: 2026-07-24T09:21:22*

---

## Raw Tool-Call Trail (audit_log)

### Step 0
**Called:**
```
import pandas as pd
print('Shape:', df.shape)
print('Columns and dtypes:')
print(df.dtypes)
print('\nFirst few rows:')
print(df.head())
print('\nNumeric description:')
print(df.describe())
```
**Result:**
```
Shape: (32561, 15)
Columns and dtypes:
age               int64
workclass           str
fnlwgt            int64
education           str
education_num     int64
marital_status      str
occupation          str
relationship        str
race                str
sex                 str
capital_gain      int64
capital_loss      int64
hours_per_week    int64
native_country      str
income              str
dtype: object

First few rows:
   age         workclass  fnlwgt  education  ...  capital_loss hours_per_week native_country income
0   39         State-gov   77516  Bachelors  ...             0             40  United-States  <=50K
1   50  Self-emp-not-inc   83311  Bachelors  ...             0             13  United-States  <=50K
2   38           Private  215646    HS-grad  ...             0             40  United-States  <=50K
3   53           Private  234721       11th  ...             0             40  United-States  <=50K
4   28           Private  338409  Bachelors  ...             0             40           Cuba  <=50K

[5 rows x 15 columns]

Numeric description:
                age        fnlwgt  education_num  capital_gain  capital_loss  hours_per_week
count  32561.000000  3.256100e+04   32561.000000  32561.000000  32561.000000    32561.000000
mean      38.581647  1.897784e+05      10.080679   1077.648844     87.303830       40.437456
std       13.640433  1.055500e+05       2.572720   7385.292085    402.960219       12.347429
min       17.000000  1.228500e+04       1.000000      0.000000      0.000000        1.000000
25%       28.000000  1.178270e+05       9.000000      0.000000      0.000000       40.000000
50%       37.000000  1.783560e+05      10.000000      0.000000      0.000000       40.000000
75%       48.000000  2.370510e+05      12.000000      0.000000      0.000000       45.000000
max       90.000000  1.484705e+06      16.000000  99999.000000   4356.000000       99.000000

```

### Step 1
**Called:**
```
missingness_report(cols=['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'])
```
**Result:**
```
age: 32561 non-null (100.0%), 0 missing (0.0%)
workclass: 30725 non-null (94.36%), 1836 missing (5.64%)
fnlwgt: 32561 non-null (100.0%), 0 missing (0.0%)
education: 32561 non-null (100.0%), 0 missing (0.0%)
education_num: 32561 non-null (100.0%), 0 missing (0.0%)
marital_status: 32561 non-null (100.0%), 0 missing (0.0%)
occupation: 30718 non-null (94.34%), 1843 missing (5.66%)
relationship: 32561 non-null (100.0%), 0 missing (0.0%)
race: 32561 non-null (100.0%), 0 missing (0.0%)
sex: 32561 non-null (100.0%), 0 missing (0.0%)
capital_gain: 32561 non-null (100.0%), 0 missing (0.0%)
capital_loss: 32561 non-null (100.0%), 0 missing (0.0%)
hours_per_week: 32561 non-null (100.0%), 0 missing (0.0%)
native_country: 31978 non-null (98.21%), 583 missing (1.79%)
income: 32561 non-null (100.0%), 0 missing (0.0%)
```

### Step 2
**Called:**
```
compute(expressions=["df['age'].nunique()", "df['fnlwgt'].nunique()", "df['education_num'].nunique()", "df['capital_gain'].nunique()", "df['capital_loss'].nunique()", "df['hours_per_week'].nunique()"])
```
**Result:**
```
df['age'].nunique() = 73
df['fnlwgt'].nunique() = 21648
df['education_num'].nunique() = 16
df['capital_gain'].nunique() = 119
df['capital_loss'].nunique() = 92
df['hours_per_week'].nunique() = 94
```

### Step 3
**Called:**
```
print('Skewness:')
print(df[ ['age','fnlwgt','education_num','capital_gain','capital_loss','hours_per_week'] ].skew())
```
**Result:**
```
Skewness:
age                0.558743
fnlwgt             1.446980
education_num     -0.311676
capital_gain      11.953848
capital_loss       4.594629
hours_per_week     0.227643
dtype: float64

```

### Step 4
**Called:**
```
compute(expressions=["df['income'].value_counts().to_dict()", "(df['income'].value_counts(normalize=True)*100).to_dict()"])
```
**Result:**
```
df['income'].value_counts().to_dict() = {'<=50K': 24720, '>50K': 7841}
(df['income'].value_counts(normalize=True)*100).to_dict() = {'<=50K': 75.91904425539757, '>50K': 24.080955744602438}
```

### Step 5
**Called:**
```
compute(expressions=['df[col].nunique()/len(df)*100 for col in df.columns'])
```
**Result:**
```
(auto-repaired unparenthesized comprehension -> [df[col].nunique()/len(df)*100 for col in df.columns])
[df[col].nunique()/len(df)*100 for col in df.columns] = [0.22419458861828567, 0.02456926998556555, 66.48444458094039, 0.0491385399711311, 0.0491385399711311, 0.021498111237369857, 0.04299622247473971, 0.018426952489174166, 0.015355793740978473, 0.006142317496391388, 0.3654678910352876, 0.28254660483400384, 0.2886889223303953, 0.12591750867602347, 0.006142317496391388]
```


---

## Ground Truth (independent computation)

```
============================================================
GROUND TRUTH (computed independently, not by the agent)
============================================================

Shape: (32561, 15)

Null counts:
age                  0
workclass         1836
fnlwgt               0
education            0
education_num        0
marital_status       0
occupation        1843
relationship         0
race                 0
sex                  0
capital_gain         0
capital_loss         0
hours_per_week       0
native_country     583
income               0
dtype: int64

Unique value counts per column (flag anything close to row count as a likely ID):
  age: 73 unique / 32561 rows
  workclass: 8 unique / 32561 rows
  fnlwgt: 21648 unique / 32561 rows
  education: 16 unique / 32561 rows
  education_num: 16 unique / 32561 rows
  marital_status: 7 unique / 32561 rows
  occupation: 14 unique / 32561 rows
  relationship: 6 unique / 32561 rows
  race: 5 unique / 32561 rows
  sex: 2 unique / 32561 rows
  capital_gain: 119 unique / 32561 rows
  capital_loss: 92 unique / 32561 rows
  hours_per_week: 94 unique / 32561 rows
  native_country: 41 unique / 32561 rows
  income: 2 unique / 32561 rows

Skewness (numeric columns only):
  age: 0.5587
  fnlwgt: 1.4469
  education_num: -0.3117
  capital_gain: 11.9533
  capital_loss: 4.5944
  hours_per_week: 0.2276

Compare these numbers by hand against the agent's audit_log entries and final report. A mismatch, or a claim with no matching tool call in audit_log, is exactly what you're here to catch.
```

---

## Automated Verification (full detail)

```
============================================================
AUTOMATED REPORT VERIFICATION
============================================================

[1] Unverified numbers (no matching tool-call output): 0

[2] Unverified quoted values/labels: 5
    ⚠ 'Min‑Max or Standard scaling' — not found in any tool output
    ⚠ 'log1p' — not found in any tool output
    ⚠ '`class_weight='balanced'` in logistic regression' — not found in any tool output
    ⚠ 'balanced' — not found in any tool output
    ⚠ 'class‑weighting, oversampling (SMOTE' — not found in any tool output

[3] Possible internal contradictions: 0

[4] Citation-tag problems: 0 serious (+ 0 minor indexing slips)

5 serious issue(s) flagged — review above (0 minor indexing slip(s) not counted toward this total)
============================================================
```