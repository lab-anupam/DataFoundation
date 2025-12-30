"""
===========================================================
Data Cleaning Pipeline
Week 2 - Day 4 | Data Foundations

Author: Anupam Bhattacharyya

Topics Covered:
- Data inspection
- Missing value handling
- Duplicate removal
- Datatype correction
- Outlier detection (IQR)
- Outlier handling using clip()
- Reusable cleaning pipeline

Purpose:
- Convert raw, messy data into clean, ML-ready data
- Follow real-world data engineering practices
===========================================================
"""

import pandas as pd

# ---------------------------------------------------------
# 1. CREATE RAW (DIRTY) SAMPLE DATA
# ---------------------------------------------------------

raw_data = {
    "order_id": [1, 2, 3, 3, 4],
    "customer": ["Alice", "Bob", None, "Bob", "Charlie"],
    "amount": ["100", "200", "150", "150", "5000"],  # strings + outlier
    "order_date": ["2024-01-01", "2024-01-02", "2024-01-02", "2024-01-02", "invalid_date"]
}

df = pd.DataFrame(raw_data)

print("\n================ RAW DATA ================")
print(df)

# ---------------------------------------------------------
# 2. INITIAL DATA INSPECTION (ALWAYS FIRST)
# ---------------------------------------------------------

print("\n================ INSPECTION ================")
df.info()

print("\nMissing values:")
print(df.isna().sum())

print("\nDuplicate rows:", df.duplicated().sum())

# ---------------------------------------------------------
# 3. DATATYPE FIXING
# ---------------------------------------------------------

# Convert amount to numeric (invalid values -> NaN)
df["amount"] = pd.to_numeric(df["amount"], errors="coerce")

# Convert order_date to datetime (invalid -> NaT)
df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

print("\n================ AFTER DATATYPE FIX ================")
df.info()

# ---------------------------------------------------------
# 4. HANDLE MISSING VALUES
# ---------------------------------------------------------

# Fill missing customer names
df["customer"] = df["customer"].fillna("Unknown")

# Fill missing amount with median
df["amount"] = df["amount"].fillna(df["amount"].median())

print("\n================ AFTER MISSING VALUE HANDLING ================")
print(df)

# ---------------------------------------------------------
# 5. REMOVE DUPLICATES
# ---------------------------------------------------------

df = df.drop_duplicates(subset=["order_id"], keep="last")

print("\n================ AFTER DUPLICATE REMOVAL ================")
print(df)

# ---------------------------------------------------------
# 6. OUTLIER DETECTION USING IQR
# ---------------------------------------------------------

Q1 = df["amount"].quantile(0.25)
Q3 = df["amount"].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("\n================ OUTLIER LIMITS ================")
print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

# ---------------------------------------------------------
# 7. HANDLE OUTLIERS USING CLIP (WINSORIZATION)
# ---------------------------------------------------------

df["amount"] = df["amount"].clip(
    lower=lower_bound,
    upper=upper_bound
)

print("\n================ AFTER OUTLIER CLIPPING ================")
print(df)

# ---------------------------------------------------------
# 8. FINAL VALIDATION
# ---------------------------------------------------------

print("\n================ FINAL VALIDATION ================")
df.info()

print("\nMissing values after cleaning:")
print(df.isna().sum())

print("\nStatistical summary:")
print(df.describe())

# ---------------------------------------------------------
# 9. CLEANING PIPELINE AS A FUNCTION (REUSABLE)
# ---------------------------------------------------------

def clean_data_pipeline(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Reusable data cleaning pipeline
    """
    df = input_df.copy()

    # Fix datatypes
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")  #“If a value cannot be converted, don’t crash — replace it with NaN.”
    df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce")

    # Handle missing values
    df["customer"] = df["customer"].fillna("Unknown")
    df["amount"] = df["amount"].fillna(df["amount"].median())

    # Remove duplicates
    df = df.drop_duplicates(subset=["order_id"], keep="last")

    # Outlier handling
    Q1 = df["amount"].quantile(0.25)
    Q3 = df["amount"].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    df["amount"] = df["amount"].clip(lower=lower, upper=upper)

    return df

print("""
===========================================================
CLEANING SUMMARY:
- Inspected raw data
- Fixed datatypes
- Handled missing values
- Removed duplicates
- Capped outliers using IQR + clip()
- Validated final dataset

KEY IDEA:
Clean data is more important than complex models.
===========================================================
""")
