import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ---  MOCK DATA GENERATION & RFM FEATURE ENGINEERING ---

np.random.seed(42)
data_size = 2000
today = datetime.datetime.now()
MAX_DAYS_AGO = 365

transactions = pd.DataFrame({
    'CustomerID': np.random.randint(100, 300, data_size),
    'PurchaseDate': [today - datetime.timedelta(days=d) for d in np.random.randint(1, MAX_DAYS_AGO, data_size)],
    'Amount': np.round(np.random.uniform(5, 500, data_size), 2)
})

# Feature Engineering: RFM (Recency, Frequency, Monetary)
rfm = transactions.groupby('CustomerID').agg(
    Recency=('PurchaseDate', lambda x: (today - x.max()).days), # Days since last purchase
    Frequency=('PurchaseDate', 'count'),                        # Total number of purchases
    Monetary=('Amount', 'sum')                                  # Total spend
).reset_index()

print(f"--- RFM Features Created for {len(rfm)} Customers ---")

# ---  DATA MODEL DEVELOPMENT (K-Means Clustering for Segmentation) ---

# Preparing data for modeling
X = rfm[['Recency', 'Frequency', 'Monetary']]

# Standardizing the features (crucial for clustering/modeling)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# A simple model: K-Means Clustering for Customer Segmentation (LTV proxy)
K = 3
kmeans = KMeans(n_clusters=K, random_state=42, n_init=10, max_iter=300)
rfm['Cluster'] = kmeans.fit_predict(X_scaled)

# Analyze Cluster Characteristics
cluster_summary = rfm.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().sort_values(by='Monetary')
cluster_summary.rename(columns={'Recency': 'Avg_Recency', 'Frequency': 'Avg_Frequency', 'Monetary': 'Avg_Monetary'}, inplace=True)

# Maping clusters to descriptive names based on their monetary value
# The lowest monetary cluster is "Low Value" (0), highest is "High Value" (2)
cluster_map = {
    cluster_summary.index[0]: 'Low_Value',
    cluster_summary.index[1]: 'Medium_Value',
    cluster_summary.index[2]: 'High_Value'
}
rfm['LTV_Segment'] = rfm['Cluster'].map(cluster_map)

# ---  VISUALIZATION AND REPORTING (Present Innovative Findings) ---

plt.figure(figsize=(8, 6))
sns.boxplot(x='LTV_Segment', y='Monetary', data=rfm.sort_values(by='Monetary'), palette='tab10')
plt.title('Customer Segmentation by Monetary Value (LTV Proxy)')
plt.ylabel('Total Monetary Value')
plt.xlabel('LTV Segment')
plt.tight_layout()
plt.show()

# --- 4. BUSINESS INSIGHTS ---
print("\n" + "="*60)
print("CUSTOMER LTV SEGMENTATION INSIGHTS (Driving Business Decisions)")
print("="*60)
print("Cluster Summary (Sorted by Monetary Value):\n", cluster_summary)
print("\nHigh Value Segment (Recommendation):")
high_value_count = len(rfm[rfm['LTV_Segment'] == 'High_Value'])
print(f" - Found {high_value_count} customers in the High Value Segment.")
print(" - Strategy: Implement a targeted loyalty program to ensure retention of these top spenders.")
print("="*60)
