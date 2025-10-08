
# 🎯 Customer LTV Segmentation using Statistical Modeling

This project focuses on the **development and implementation of data models** by applying unsupervised machine learning (K-Means Clustering) to customer transaction data to segment them based on potential **Lifetime Value (LTV)**. This provides innovative insights to drive targeted business strategies.

## 🚀 Project Overview

The goal is to move beyond descriptive statistics and use predictive modeling techniques to group customers into segments (Low, Medium, High Value) based on their purchase behavior (Recency, Frequency, Monetary).

### Key Features :

* **Data Model Development:** Implements the K-Means clustering algorithm using Scikit-learn for customer segmentation, a proxy for **Statistical Modeling** and LTV prediction.
* **Feature Engineering:** Demonstrates proficiency in creating derived features (RFM metrics) from transactional data, simulating **database support** and data maintenance for analytical needs.
* **Strategy and Innovation:** The resulting segments lead to clear, actionable **business strategies** (e.g., retention for High Value, re-engagement for Low Value).
* **Data Preparation:** Utilizes `StandardScaler` to ensure features are appropriately scaled for model training, a best practice in data science.

## 🛠️ Technology Stack

* **Language:** Python
* **Libraries:**
    * `Scikit-learn`: For the implementation of the K-Means clustering **data model**.
    * `Pandas` & `NumPy`: For data preparation and RFM feature engineering.
    * `Matplotlib` & `Seaborn`: For visualizing cluster characteristics and segment distribution.

## ⚙️ Installation and Execution

### 1. Install Dependencies

Open your terminal and install the necessary Python libraries:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
2. Execution
Save the project code into a file named customer_ltv_segmenter.py.

Run the script from your terminal:
python customer_ltv_segmenter.py

The script will generate the RFM features, train the segmentation model, display a box plot visualization of the segments, and provide a business strategy recommendation.
