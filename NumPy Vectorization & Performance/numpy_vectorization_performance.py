"""
===========================================================
NumPy Vectorization & Performance
Week 2 - Day 3 | Data Foundations

Author: Anupam Bhattacharyya

Topics Covered:
- What is vectorization
- NumPy arrays
- Element-wise operations
- Boolean masking
- np.where
- Aggregations
- Broadcasting
- Vectorization vs loops

Purpose:
- Write fast, ML-ready numerical code
- Avoid Python loops
- Understand performance mindset
===========================================================
"""

import numpy as np

# ---------------------------------------------------------
# 1. PYTHON LOOP VS NUMPY VECTORIZATION
# ---------------------------------------------------------

numbers = [1, 2, 3, 4, 5]

# ❌ Python loop (slow)
squares_loop = []
for x in numbers:
    squares_loop.append(x * x)

print("Squares using Python loop:", squares_loop)

# ✅ NumPy vectorized (fast)
arr = np.array(numbers)
squares_vectorized = arr * arr

print("Squares using NumPy vectorization:", squares_vectorized)

# ---------------------------------------------------------
# 2. NUMPY ARRAY CREATION
# ---------------------------------------------------------

print("\n=== ARRAY CREATION ===")
print(np.array([1, 2, 3]))
print(np.zeros(5))
print(np.ones(5))
print(np.arange(0, 10))
print(np.linspace(0, 1, 5))

# ---------------------------------------------------------
# 3. ARRAY PROPERTIES
# ---------------------------------------------------------

print("\n=== ARRAY PROPERTIES ===")
print("Shape:", arr.shape)
print("Dimensions:", arr.ndim)
print("Size:", arr.size)

# ---------------------------------------------------------
# 4. ELEMENT-WISE OPERATIONS
# ---------------------------------------------------------

a = np.array([10, 20, 30])
b = np.array([1, 2, 3])

print("\n=== ELEMENT-WISE OPERATIONS ===")
print("Addition:", a + b)
print("Subtraction:", a - b)
print("Multiplication:", a * b)
print("Division:", a / b)

# ---------------------------------------------------------
# 5. BOOLEAN MASKING (VERY IMPORTANT)
# ---------------------------------------------------------

values = np.array([10, 20, 30, 40, 50])

print("\n=== BOOLEAN MASKING ===")
print("Values > 25:", values[values > 25])

# ---------------------------------------------------------
# 6. CONDITIONAL LOGIC WITH np.where
# ---------------------------------------------------------

labels = np.where(values > 30, "HIGH", "LOW")

print("\n=== CONDITIONAL LOGIC (np.where) ===")
print(labels)

# ---------------------------------------------------------
# 7. AGGREGATION FUNCTIONS
# ---------------------------------------------------------

print("\n=== AGGREGATIONS ===")
print("Sum:", values.sum())
print("Mean:", values.mean())
print("Min:", values.min())
print("Max:", values.max())
print("Std Dev:", values.std())

# ---------------------------------------------------------
# 8. BROADCASTING (MAGIC CONCEPT)
# ---------------------------------------------------------

print("\n=== BROADCASTING ===")
print("Original:", a)
print("Add 5:", a + 5)
print("Multiply by 2:", a * 2)

# ---------------------------------------------------------
# 9. VECTORIZATION IN REAL-WORLD STYLE
# ---------------------------------------------------------

prices = np.array([100, 200, 300])
tax_rate = 0.18

final_prices = prices * (1 + tax_rate)

print("\n=== REAL-WORLD EXAMPLE ===")
print("Prices:", prices)
print("Final prices with tax:", final_prices)

# ---------------------------------------------------------
# 10. AVOID LOOPS — COMPARISON
# ---------------------------------------------------------

# ❌ Loop-based approach
adjusted_loop = []
for price in prices:
    adjusted_loop.append(price * 1.18)

# ✅ Vectorized approach
adjusted_vectorized = prices * 1.18

print("\nLoop-based result:", adjusted_loop)
print("Vectorized result:", adjusted_vectorized)

# ---------------------------------------------------------
# 11. INTERVIEW-READY SUMMARY
# ---------------------------------------------------------

print("""
===========================================================
SUMMARY:
- NumPy avoids Python loops
- Operations are vectorized and fast
- Boolean masking replaces filtering loops
- Broadcasting enables scalar-array operations
- NumPy is the foundation of Pandas & ML libraries

Golden Rule:
❌ Avoid loops
✅ Think in arrays
===========================================================
""")
