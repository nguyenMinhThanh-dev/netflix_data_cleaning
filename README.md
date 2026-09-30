# 🎬 Netflix Data Cleaning & Feature Engineering

A Python data cleaning pipeline that resolves missing values using relational inference, normalizes text fields, and extracts temporal features from the Netflix Titles dataset for EDA and visualization.

## 📁 Repository Structure
```text
.
├── dataset/
│   ├── netflix_titles.csv      # Raw dataset
│   └── netflix_cleaned.csv     # Processed dataset
├── script/
│   └── cleaning.py             # Data cleaning script
└── README.md
🛠️ Key Data Cleaning Steps
Relational Imputation: Filled missing director and country values using an actor-director collaboration matrix and director-country mappings.

Text Normalization: Extracted the primary country for multi-national titles.

Feature Extraction: Parsed date_added to datetime and extracted year_added and month_added.

Data Cleanup: Filled missing cast with 'Unknown', dropped invalid records (date_added, rating, duration), and removed the unused description column.
