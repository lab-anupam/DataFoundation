"""
===========================================================
Pandas I/O & Data Inspection
Week 2 - Day 1 | Data Foundations

Author: Anupam Bhattacharyya
Purpose:
- Demonstrate real-world pandas read_csv usage
- Perform thorough data inspection & validation
- Serve as a reusable inspection template for projects
===========================================================
"""

import pandas as pd

# ---------------------------------------------------------
# 1. SAMPLE DATA CREATION (for demo purpose)
# ---------------------------------------------------------
# In real projects, data comes from CSV / JSON / APIs

sample_data = {
    "order_id": [1, 2, 3, 3],
    "customer": ["Alice", "Bob", None, "Bob"],
    "amount": [250, 400, 150, -50],
    "order_date": ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-02"]
}

df = pd.DataFrame(sample_data)

# Save sample CSV
df.to_csv("sales.csv", index=False)

# ---------------------------------------------------------
# 2. READING CSV WITH PRODUCTION-GRADE OPTIONS
# ---------------------------------------------------------
df = pd.read_csv(
    "sales.csv",
    sep=",",                       # Column separator
    encoding="utf-8",              # Text encoding
    parse_dates=["order_date"],    # Convert to datetime
    na_values=["", "NULL", "NA"]   # Treat these as NaN
)

print("\n=== DATA LOADED SUCCESSFULLY ===")

# ---------------------------------------------------------
# 3. BASIC DATA INSPECTION
# ---------------------------------------------------------
print("\n--- HEAD (First 5 rows) ---")
print(df.head())

print("\n--- TAIL (Last 5 rows) ---")
print(df.tail())

print("\n--- RANDOM SAMPLE ---")
print(df.sample(2))

# ---------------------------------------------------------
# 4. SHAPE & SIZE
# ---------------------------------------------------------
print("\n--- SHAPE & SIZE ---")
print("Rows, Columns:", df.shape)
print("Total cells:", df.size)

# ---------------------------------------------------------
# 5. SCHEMA & DATA TYPES
# ---------------------------------------------------------
print("\n--- INFO (Schema Inspection) ---")
df.info()

print("\n--- COLUMN DATA TYPES ---")
print(df.dtypes)

# ---------------------------------------------------------
# 6. DESCRIPTIVE STATISTICS
# ---------------------------------------------------------
print("\n--- NUMERICAL SUMMARY ---")
print(df.describe())

print("\n--- CATEGORICAL SUMMARY ---")
print(df.describe(include="object"))

# ---------------------------------------------------------
# 7. MISSING DATA ANALYSIS
# ---------------------------------------------------------
print("\n--- MISSING VALUES (COUNT) ---")
print(df.isna().sum())

print("\n--- MISSING VALUES (%) ---")
print(df.isna().mean() * 100)

print("\n--- ROWS WITH ANY MISSING VALUE ---")
print(df[df.isna().any(axis=1)])

# ---------------------------------------------------------
# 8. DUPLICATE ANALYSIS
# ---------------------------------------------------------
print("\n--- DUPLICATE ROW COUNT ---")
print(df.duplicated().sum())

print("\n--- DUPLICATE ROWS ---")
print(df[df.duplicated()])

# ---------------------------------------------------------
# 9. UNIQUENESS & VALUE DISTRIBUTION
# ---------------------------------------------------------
print("\n--- UNIQUE VALUES PER COLUMN ---")
print(df.nunique())

print("\n--- VALUE COUNTS (customer) ---")
print(df["customer"].value_counts(dropna=False))

# ---------------------------------------------------------
# 10. RANGE & ANOMALY CHECKS
# ---------------------------------------------------------
print("\n--- AMOUNT RANGE ---")
print("Min:", df["amount"].min())
print("Max:", df["amount"].max())

print("\n--- NEGATIVE AMOUNTS (Potential Data Issue) ---")
print(df[df["amount"] < 0])

# ---------------------------------------------------------
# 11. QUICK DATA PROFILING SUMMARY
# ---------------------------------------------------------
profile_summary = {
    "total_rows": len(df),
    "total_columns": len(df.columns),
    "missing_cells": df.isna().sum().sum(),
    "duplicate_rows": df.duplicated().sum(),
    "negative_amounts": (df["amount"] < 0).sum()
}

print("\n--- DATA PROFILING SUMMARY ---")
for key, value in profile_summary.items():
    print(f"{key}: {value}")

# ---------------------------------------------------------
# 12. INTERVIEW-READY FINAL STATEMENT
# ---------------------------------------------------------
print("""
INSPECTION CONCLUSION:
- Dataset contains missing values
- Duplicate rows detected
- Negative transaction values present
- Data requires cleaning before analytics or ML usage
""")
