"""
===========================================================
Pandas Advanced Operations
Week 2 - Day 2 | Data Foundations

Author: Anupam Bhattacharyya

Topics Covered:
- groupby()
- agg()
- transform()
- merge()
- join()
- window functions (shift, rolling)

Purpose:
- Build strong SQL-like thinking in Pandas
- Prepare for real-world analytics & ML pipelines
===========================================================
"""

import pandas as pd

# ---------------------------------------------------------
# 1. SAMPLE DATA CREATION
# ---------------------------------------------------------

sales_data = {
    "customer": ["Alice", "Bob", "Alice", "Bob", "Charlie"],
    "product": ["A", "A", "B", "B", "A"],
    "amount": [100, 200, 150, 300, 400]
}

df = pd.DataFrame(sales_data)

print("\n=== ORIGINAL DATA ===")
print(df)

# ---------------------------------------------------------
# 2. GROUPBY BASICS
# ---------------------------------------------------------
# Group data and calculate metrics per group

print("\n=== TOTAL SALES PER CUSTOMER ===")
print(df.groupby("customer")["amount"].sum())

print("\n=== MULTIPLE AGGREGATIONS ===")
print(df.groupby("customer")["amount"].agg(["sum", "mean", "count"]))

print("\n=== GROUP BY MULTIPLE COLUMNS ===")
print(df.groupby(["customer", "product"])["amount"].sum())

# Reset index to get a flat table
print("\n=== GROUPBY WITH RESET INDEX ===")
print(
    df.groupby("customer")["amount"]
      .sum()
      .reset_index()
)

# ---------------------------------------------------------
# 3. AGG() — CLEAN & SCALABLE AGGREGATION
# ---------------------------------------------------------

print("\n=== AGG WITH NAMED COLUMNS ===")
agg_df = df.groupby("customer").agg(
    total_sales=("amount", "sum"),
    average_sales=("amount", "mean"),
    order_count=("amount", "count")
)

print(agg_df)

# ---------------------------------------------------------
# 4. TRANSFORM() — GROUP-WISE BUT ROW-LEVEL
# ---------------------------------------------------------
# Same shape output as original dataframe

print("\n=== TRANSFORM EXAMPLE ===")

df["customer_avg_sale"] = df.groupby("customer")["amount"].transform("mean")

print(df)

# ---------------------------------------------------------
# 5. MERGE() — SQL STYLE JOINS
# ---------------------------------------------------------

orders = pd.DataFrame({
    "order_id": [1, 2, 3],
    "customer_id": [101, 102, 103],
    "amount": [250, 300, 150]
})

customers = pd.DataFrame({
    "customer_id": [101, 102, 104],
    "customer_name": ["Alice", "Bob", "David"]
})

print("\n=== ORDERS TABLE ===")
print(orders)

print("\n=== CUSTOMERS TABLE ===")
print(customers)

print("\n=== INNER JOIN ===")
print(
    orders.merge(customers, on="customer_id", how="inner")
)

print("\n=== LEFT JOIN (MOST COMMON) ===")
print(
    orders.merge(customers, on="customer_id", how="left")
)

print("\n=== RIGHT JOIN ===")
print(
    orders.merge(customers, on="customer_id", how="right")
)

print("\n=== OUTER JOIN ===")
print(
    orders.merge(customers, on="customer_id", how="outer")
)

# ---------------------------------------------------------
# 6. JOIN() — INDEX BASED JOIN
# ---------------------------------------------------------

print("\n=== INDEX-BASED JOIN ===")
print(
    orders.set_index("customer_id")
          .join(customers.set_index("customer_id"), how="left")
)

# ---------------------------------------------------------
# 7. WINDOW FUNCTIONS
# ---------------------------------------------------------
# Used for trends, time-series, comparisons

print("\n=== SHIFT (PREVIOUS ROW) ===")
df["previous_amount"] = df["amount"].shift(1)
print(df)

print("\n=== DIFFERENCE BETWEEN ROWS ===")
df["amount_change"] = df["amount"] - df["amount"].shift(1)
print(df)

print("\n=== ROLLING AVERAGE ===")
df["rolling_avg_2"] = df["amount"].rolling(window=2).mean()
print(df)

# ---------------------------------------------------------
# 8. GROUP-WISE WINDOW FUNCTION
# ---------------------------------------------------------

print("\n=== GROUP-WISE ROLLING AVERAGE ===")
df["customer_rolling_avg"] = (
    df.groupby("customer")["amount"]
      .rolling(window=2)
      .mean()
      .reset_index(level=0, drop=True)
)

print(df)

# ---------------------------------------------------------
# 9. INTERVIEW-READY SUMMARY
# ---------------------------------------------------------

print("""
===========================================================
SUMMARY:
- groupby() → metrics per group
- agg() → clean multi-metric calculations
- transform() → feature engineering
- merge() → relational joins (SQL style)
- join() → index-based joins
- shift & rolling → window / trend analysis

These operations are core to analytics, ML features,
and data engineering pipelines.
===========================================================
""")
