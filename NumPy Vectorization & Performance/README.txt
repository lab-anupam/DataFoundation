===========================================================
NUMPY VECTORIZATION & PERFORMANCE
Week 2 – Day 3
===========================================================

Author: Anupam Bhattacharyya

This README explains the core NumPy concepts required for
high-performance numerical computing, analytics, and
machine learning pipelines.

The focus is on writing FAST, CLEAN, and SCALABLE code by
avoiding Python loops and using vectorized operations.

-----------------------------------------------------------
WHAT IS NUMPY?
-----------------------------------------------------------

NumPy (Numerical Python) is a foundational Python library
for numerical computation.

Key features:
- Fast multi-dimensional arrays
- Vectorized mathematical operations
- Efficient memory usage
- Backbone of Pandas, SciPy, scikit-learn, TensorFlow, PyTorch

-----------------------------------------------------------
WHY NUMPY IS IMPORTANT
-----------------------------------------------------------

- Python loops are slow for large data
- NumPy executes operations in optimized C code
- Used in data engineering, analytics, ML, and AI systems
- Essential for performance-critical applications

Golden rule:
❌ Avoid Python loops
✅ Think in arrays

-----------------------------------------------------------
KEY CONCEPTS COVERED
-----------------------------------------------------------

1. Vectorization
2. NumPy arrays
3. Element-wise operations
4. Boolean masking
5. Conditional logic using np.where
6. Aggregation functions
7. Broadcasting
8. Performance mindset

-----------------------------------------------------------
1. VECTORIZATION
-----------------------------------------------------------

Vectorization means performing operations on entire arrays
at once instead of looping element by element.

Example:
- Loop-based: slow, interpreted
- NumPy-based: fast, compiled

Why it matters:
- Cleaner code
- Better performance
- Scales to large datasets

-----------------------------------------------------------
2. NUMPY ARRAYS
-----------------------------------------------------------

NumPy arrays are:
- Homogeneous (same data type)
- Stored in contiguous memory
- Faster than Python lists for math operations

Common ways to create arrays:
- array
- zeros
- ones
- arange
- linspace

-----------------------------------------------------------
3. ELEMENT-WISE OPERATIONS
-----------------------------------------------------------

NumPy operations are element-wise by default.

Examples:
- Addition
- Subtraction
- Multiplication
- Division

Each operation is applied to corresponding elements
without explicit loops.

-----------------------------------------------------------
4. BOOLEAN MASKING
-----------------------------------------------------------

Boolean masking is used to filter data efficiently.

Concept:
- Create a boolean condition
- Use it to select values from the array

This replaces conditional loops and improves performance.

-----------------------------------------------------------
5. CONDITIONAL LOGIC (np.where)
-----------------------------------------------------------

np.where allows conditional assignment in vectorized form.

Use cases:
- Feature engineering
- Categorization
- Threshold-based labeling

Example logic:
IF condition is true → value A
ELSE → value B

-----------------------------------------------------------
6. AGGREGATION FUNCTIONS
-----------------------------------------------------------

NumPy provides fast aggregation functions such as:
- sum
- mean
- min
- max
- standard deviation

These are used for:
- Statistical analysis
- Data summaries
- ML preprocessing

-----------------------------------------------------------
7. BROADCASTING
-----------------------------------------------------------

Broadcasting allows operations between arrays of
different shapes.

Example:
- Adding a scalar to an array
- Multiplying an array by a constant

NumPy automatically expands dimensions when possible.

-----------------------------------------------------------
8. PERFORMANCE MINDSET
-----------------------------------------------------------

Preferred approach:
- Use NumPy vectorized operations
- Avoid loops and .apply()
- Use boolean masking instead of filtering loops

Performance comparison:
- Python loops → Slow
- List comprehensions → Medium
- NumPy vectorization → Fast

-----------------------------------------------------------
REAL-WORLD USE CASES
-----------------------------------------------------------

- Feature scaling
- Financial calculations
- Scientific computing
- ML model preprocessing
- Simulation and modeling

-----------------------------------------------------------
INTERVIEW-READY STATEMENTS
-----------------------------------------------------------

- "NumPy is faster because it uses vectorized operations
   implemented in optimized C code."

- "Broadcasting allows operations between arrays of
   different shapes without explicit looping."

- "I prefer NumPy vectorization over Python loops for
   performance-critical computations."

-----------------------------------------------------------
HOW TO RUN THE CODE
-----------------------------------------------------------

1. Install dependencies:
   pip install numpy

2. Run the script:
   python numpy_vectorization_performance.py

-----------------------------------------------------------
SUMMARY
-----------------------------------------------------------

NumPy is essential for:
- High-performance computation
- Clean and scalable code
- Data science and machine learning foundations

Key takeaway:
NO LOOPS. THINK IN ARRAYS.

-----------------------------------------------------------
END OF README
-----------------------------------------------------------
