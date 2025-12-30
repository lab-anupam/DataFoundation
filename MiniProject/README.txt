===========================================================
WEEK 2 – DAY 7
END-TO-END EDA PIPELINE PROJECT
===========================================================

Author: Anupam Bhattacharyya
Focus Area: Data Foundations
Project Type: Exploratory Data Analysis (EDA) Pipeline

===========================================================
PROJECT OVERVIEW
===========================================================

This project demonstrates a complete, real-world Exploratory
Data Analysis (EDA) pipeline.

The goal is to take raw, imperfect data and transform it into
a clean, well-understood, and ML-ready dataset by applying
systematic inspection, cleaning, analysis, and validation.

This project reflects how EDA is performed in industry before
any machine learning model is built.

===========================================================
BUSINESS CONTEXT
===========================================================

The dataset represents customer transaction and behavior data
with a churn outcome.

Each row corresponds to a customer, and the objective is to
understand patterns that may explain customer churn.

The project focuses on:
- Understanding data quality
- Detecting anomalies and outliers
- Identifying meaningful patterns
- Preventing data leakage

===========================================================
DATASET DESCRIPTION
===========================================================

Columns used in the dataset:

- customer_id      : Unique identifier for each customer
- age              : Age of the customer (contains missing and invalid values)
- category         : Product or service category
- tenure_months    : Number of months the customer has stayed
- monthly_spend    : Monthly spending amount (contains outliers)
- total_spend      : Total lifetime spend (potential leakage feature)
- churn            : Target variable (1 = churned, 0 = retained)

The dataset intentionally includes:
- Missing values
- Invalid values (e.g., age > 100)
- Outliers
- A leakage-prone feature

===========================================================
PROJECT STRUCTURE
===========================================================

week2_day7_eda_pipeline/
│
├── eda_pipeline.py        → Runnable EDA script
├── eda_pipeline.ipynb     → Notebook-style EDA analysis
├── README.txt             → Project documentation (this file)
└── requirements.txt       → Required Python libraries

===========================================================
EDA PIPELINE STEPS
===========================================================

1. Data Loading
----------------
Raw data is loaded (simulated in code to resemble production
data ingestion).

2. Initial Inspection
---------------------
The following checks are performed:
- df.info()
- df.describe()
- Missing value counts

This step ensures understanding of schema, datatypes, and
basic statistical properties before any transformation.

3. Data Cleaning
----------------
Cleaning decisions are made conservatively and logically:

- Numeric conversion using errors="coerce"
- Missing age values filled using median
- Invalid age values (>100) corrected using business rules
- Outliers in monthly_spend handled using IQR-based clipping

The goal is to fix data quality issues without over-cleaning
or distorting the data.

4. Exploratory Data Analysis (EDA)
----------------------------------
The following analyses are performed:

- Distribution analysis (histograms, boxplots)
- Categorical distribution analysis
- Feature vs target analysis
- Crosstab analysis for category vs churn
- Correlation analysis

EDA is used to understand:
- Data skewness
- Outlier behavior
- Category imbalance
- Feature relevance

5. Data Leakage Detection
-------------------------
Correlation analysis is used to identify suspicious features.

The column "total_spend" shows very high correlation with the
target variable and is identified as data leakage because:

- It includes future information
- It would not be available at prediction time

This column is explicitly removed from the dataset.

6. Final Validation
-------------------
After cleaning and leakage removal, the dataset is validated:

- No unexpected missing values
- Correct datatypes
- Reasonable statistical ranges
- Safe for ML modeling

===========================================================
KEY INSIGHTS
===========================================================

- Monthly spend shows right-skewed distribution with outliers
- Certain categories show higher churn proportions
- High correlation does not always imply usefulness
- Data leakage can dramatically inflate model performance
- Proper EDA prevents misleading ML results

===========================================================
TOOLS & TECHNOLOGIES USED
===========================================================

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn

===========================================================
HOW TO RUN THE PROJECT
===========================================================

1. Install dependencies:
   pip install pandas numpy matplotlib seaborn

2. Run the script:
   python eda_pipeline.py

3. Or open the notebook:
   jupyter notebook eda_pipeline.ipynb

===========================================================
INTERVIEW-READY SUMMARY
===========================================================

This project demonstrates the ability to:
- Perform structured EDA
- Clean real-world data safely
- Detect and prevent data leakage
- Analyze feature–target relationships
- Prepare datasets for ML readiness

INTERVIEW STATEMENT:
"I built an end-to-end EDA pipeline that inspects raw data,
applies safe cleaning, performs exploratory analysis, detects
data leakage, and validates the dataset for ML modeling."

===========================================================
KEY TAKEAWAY
===========================================================

Clean and well-understood data is more important than complex
models.

EDA is not optional — it is foundational.

===========================================================
END OF README
===========================================================
