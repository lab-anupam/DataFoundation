"""
===========================================================
EDA PIPELINE PROJECT
Week 2 – Day 7 | Data Foundations

Author: Anupam Bhattacharyya

Goal:
- Load raw data
- Inspect schema & data quality
- Clean data safely
- Perform EDA
- Detect data leakage
- Prepare ML-ready dataset

This script represents a REAL end-to-end EDA workflow.
===========================================================
"""

# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# =========================================================
# 2. LOAD / CREATE RAW DATA
# =========================================================
# In real projects, this would be pd.read_csv(...)
# Here we simulate raw production-like data

raw_data = {
    "customer_id": [1,2,3,4,5,6,7,8,9],
    "age": [25, 30, None, 45, 200, 35, 40, None, 28],
    "category": ["A", "B", "A", "C", "B", "C", "C", "A", "B"],
    "tenure_months": [2, 5, 8, 12, 1, 36, 48, 3, 6],
    "monthly_spend": [100, 150, 120, 200, 3000, 180, 220, 130, 160],
    "total_spend": [200, 750, 960, 2400, 3000, 6480, 10560, 390, 960],
    "churn": [1, 0, 0, 0, 1, 0, 0, 1, 0]
}

df = pd.DataFrame(raw_data)

print("\n================ RAW DATA ================")
print(df)

# =========================================================
# 3. INITIAL DATA INSPECTION (MANDATORY)
# =========================================================

print("\n================ DATA INFO ================")
df.info()

print("\n================ STATISTICAL SUMMARY ================")
print(df.describe())

print("\n================ MISSING VALUES ================")
print(df.isna().sum())

# =========================================================
# 4. DATA CLEANING PIPELINE
# =========================================================

print("\n================ CLEANING DATA ================")

# ---- Fix numeric columns safely
df["age"] = pd.to_numeric(df["age"], errors="coerce")

# ---- Handle missing values
df["age"] = df["age"].fillna(df["age"].median())

# ---- Fix impossible values (business rule)
df.loc[df["age"] > 100, "age"] = df["age"].median()

# ---- Outlier handling for monthly_spend using IQR + clip
Q1 = df["monthly_spend"].quantile(0.25)
Q3 = df["monthly_spend"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

df["monthly_spend"] = df["monthly_spend"].clip(
    lower=lower_bound,
    upper=upper_bound
)

print("\nData after cleaning:")
print(df)

# =========================================================
# 5. EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================

print("\n================ EDA: DISTRIBUTIONS ================")

plt.figure()
df["monthly_spend"].hist(bins=20)
plt.title("Monthly Spend Distribution")
plt.xlabel("Monthly Spend")
plt.ylabel("Frequency")
plt.show()

print("\n================ EDA: CATEGORICAL ANALYSIS ================")
print(df["category"].value_counts(normalize=True) * 100)

plt.figure()
sns.countplot(data=df, x="category")
plt.title("Category Distribution")
plt.show()

print("\n================ EDA: FEATURE vs TARGET ================")
print("\nAverage Monthly Spend by Churn:")
print(df.groupby("churn")["monthly_spend"].mean())

print("\nCategory vs Churn:")
print(pd.crosstab(df["category"], df["churn"], normalize="index"))

# =========================================================
# 6. CORRELATION & LEAKAGE CHECK
# =========================================================

print("\n================ CORRELATION MATRIX ================")
corr = df.corr(numeric_only=True)
print(corr)

plt.figure(figsize=(6,4))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("\n================ LEAKAGE CHECK ================")
print(corr["churn"].sort_values(ascending=False))

print("""
INSIGHT:
If a feature shows extremely high correlation with churn
and is not available at prediction time, it is leakage.
""")

# ---- Remove leakage column
print("\nRemoving leakage column: total_spend")
df = df.drop(columns=["total_spend"])

# =========================================================
# 7. FINAL VALIDATION (ML-READY CHECK)
# =========================================================

print("\n================ FINAL DATASET ================")
df.info()

print("\nMissing values after cleaning:")
print(df.isna().sum())

print("\nFinal statistical summary:")
print(df.describe())

# =========================================================
# 8. FINAL SUMMARY
# =========================================================

print("""
===========================================================
EDA PIPELINE COMPLETED SUCCESSFULLY

✔ Raw data inspected
✔ Missing & invalid values handled
✔ Outliers capped safely
✔ Distributions & categorical behavior analyzed
✔ Feature-target relationships explored
✔ Data leakage detected & removed
✔ Dataset validated for ML readiness

KEY TAKEAWAY:
Always understand and validate data before modeling.
===========================================================
""")
