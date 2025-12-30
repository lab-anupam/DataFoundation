"""
===========================================================
Exploratory Data Analysis (EDA) Techniques
Week 2 - Day 5 | Data Foundations

Author: Anupam Bhattacharyya

Notebook-style script covering:
- Distributions
- Categorical analysis
- Correlation analysis
- Leakage detection
- Feature vs target analysis
- Visual intuition

Purpose:
- Understand data BEFORE modeling
- Detect issues early (skew, leakage, imbalance)
- Build intuition for ML readiness
===========================================================
"""

# =========================================================
# 1. IMPORT LIBRARIES
# =========================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# For clean plots
sns.set(style="whitegrid")

# =========================================================
# 2. CREATE SAMPLE DATASET (SIMULATING REAL DATA)
# =========================================================

data = {
    "customer_id": [1, 2, 3, 4, 5, 6, 7, 8],
    "amount": [100, 120, 130, 150, 180, 200, 2500, 3000],  # outliers
    "category": ["A", "A", "B", "B", "A", "C", "C", "C"],
    "tenure_months": [2, 5, 8, 12, 24, 36, 40, 48],
    "target": [0, 0, 0, 1, 0, 1, 1, 1]  # churn / default / conversion
}

df = pd.DataFrame(data)

print("\n================ DATASET PREVIEW ================")
print(df)

# =========================================================
# 3. BASIC DATA OVERVIEW
# =========================================================

print("\n================ DATA INFO ================")
df.info()

print("\n================ STATISTICAL SUMMARY ================")
print(df.describe())

# =========================================================
# 4. DISTRIBUTION ANALYSIS (NUMERICAL)
# =========================================================

print("\n================ DISTRIBUTION ANALYSIS ================")

# Histogram
plt.figure()
df["amount"].hist(bins=20)
plt.title("Distribution of Amount")
plt.xlabel("Amount")
plt.ylabel("Frequency")
plt.show()

# Boxplot (outlier detection)
plt.figure()
sns.boxplot(x=df["amount"])
plt.title("Boxplot of Amount")
plt.show()

print("""
INSIGHTS TO LOOK FOR:
- Is the data skewed?
- Are there extreme values?
- Does mean differ from median?
""")

# =========================================================
# 5. CATEGORICAL ANALYSIS
# =========================================================

print("\n================ CATEGORICAL ANALYSIS ================")

print("\nCategory counts:")
print(df["category"].value_counts())

print("\nCategory percentage:")
print(df["category"].value_counts(normalize=True) * 100)

# Bar plot
plt.figure()
sns.countplot(data=df, x="category")
plt.title("Category Distribution")
plt.show()

print("""
INSIGHTS TO LOOK FOR:
- Dominant categories
- Rare categories
- Class imbalance
""")

# =========================================================
# 6. CORRELATION ANALYSIS
# =========================================================

print("\n================ CORRELATION ANALYSIS ================")

corr_matrix = df.corr(numeric_only=True)
print(corr_matrix)

# Heatmap
plt.figure(figsize=(6, 4))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("""
INTERPRETATION:
- High correlation between features → redundancy
- Very high correlation with target → potential leakage
""")

# =========================================================
# 7. LEAKAGE CHECKS (CRITICAL STEP)
# =========================================================

print("\n================ LEAKAGE CHECK ================")

target_corr = corr_matrix["target"].sort_values(ascending=False)
print(target_corr)

print("""
QUESTIONS TO ASK:
- Is any feature too perfectly correlated with target?
- Would this feature be available at prediction time?
- Was this feature computed using future information?
""")

# =========================================================
# 8. FEATURE VS TARGET ANALYSIS
# =========================================================

print("\n================ FEATURE vs TARGET ================")

# Numerical vs target
print("\nAverage amount by target:")
print(df.groupby("target")["amount"].mean())

# Categorical vs target
print("\nCategory vs target distribution:")
print(pd.crosstab(df["category"], df["target"], normalize="index"))

# Visualization
plt.figure()
sns.boxplot(data=df, x="target", y="amount")
plt.title("Amount vs Target")
plt.show()

print("""
INSIGHTS:
- Does amount differ significantly by target?
- Do certain categories dominate a target class?
""")

# =========================================================
# 9. EDA CHECKLIST (MENTAL MODEL)
# =========================================================

print("""
================ EDA CHECKLIST ================

✔ Checked numerical distributions
✔ Identified skewness and outliers
✔ Analyzed categorical balance
✔ Examined correlations
✔ Checked for data leakage
✔ Studied feature–target relationships

RULE:
Never move to modeling without completing EDA.
""")

# =========================================================
# 10. FINAL INTERVIEW-READY SUMMARY
# =========================================================

print("""
===========================================================
EDA SUMMARY:
- Distributions reveal skewness and outliers
- Categorical analysis exposes imbalance
- Correlation highlights redundancy and leakage
- Feature-target analysis shows predictive power

KEY TAKEAWAY:
EDA helps you understand data behavior before modeling,
preventing silent failures and misleading accuracy.
===========================================================
""")
