import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_theme(style="whitegrid")

# Load dataset
file_path = 'src/temp/fastfood.csv'
try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded data from {file_path}")
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit(1)

# Ensure output directory exists (it should, but good practice)
output_dir = 'src/temp'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# --- Analysis 1: Average Calories by Restaurant ---
print("\n--- Analysis 1: Average Calories by Restaurant ---")
avg_calories = df.groupby('restaurant')['calories'].mean().sort_values(ascending=False).reset_index()
print(avg_calories)

plt.figure(figsize=(12, 6))
sns.barplot(data=avg_calories, x='restaurant', y='calories', palette='viridis', hue='restaurant', legend=False)
plt.title('Average Calories per Menu Item by Restaurant', fontsize=16)
plt.xlabel('Restaurant', fontsize=12)
plt.ylabel('Average Calories', fontsize=12)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{output_dir}/avg_calories_by_restaurant.png')
print(f"Saved plot to {output_dir}/avg_calories_by_restaurant.png")

# --- Analysis 2: Calories vs Total Fat Correlation ---
print("\n--- Analysis 2: Calories vs Total Fat Correlation ---")
correlation = df['calories'].corr(df['total_fat'])
print(f"Correlation between Calories and Total Fat: {correlation:.2f}")

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='calories', y='total_fat', hue='restaurant', alpha=0.7)
plt.title(f'Calories vs Total Fat (Correlation: {correlation:.2f})', fontsize=16)
plt.xlabel('Calories', fontsize=12)
plt.ylabel('Total Fat (g)', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f'{output_dir}/calories_vs_fat.png')
print(f"Saved plot to {output_dir}/calories_vs_fat.png")

# --- Analysis 3: Top 10 Items with Highest Sodium ---
print("\n--- Analysis 3: Top 10 Items with Highest Sodium ---")
top_sodium = df.nlargest(10, 'sodium')[['restaurant', 'item', 'sodium']]
print(top_sodium)

plt.figure(figsize=(12, 8))
sns.barplot(data=top_sodium, x='sodium', y='item', hue='restaurant', dodge=False)
plt.title('Top 10 Items with Highest Sodium Content', fontsize=16)
plt.xlabel('Sodium (mg)', fontsize=12)
plt.ylabel('Menu Item', fontsize=12)
plt.tight_layout()
plt.savefig(f'{output_dir}/top_10_sodium.png')
print(f"Saved plot to {output_dir}/top_10_sodium.png")

# --- Summary Insights ---
print("\n--- Summary Insights ---")
highest_cal_chain = avg_calories.iloc[0]['restaurant']
lowest_cal_chain = avg_calories.iloc[-1]['restaurant']
print(f"1. The restaurant chain with the highest average calories per item is {highest_cal_chain}.")
print(f"2. The restaurant chain with the lowest average calories per item is {lowest_cal_chain}.")
print(f"3. There is a strong positive correlation ({correlation:.2f}) between calories and total fat, indicating that higher fat content is a major driver of caloric density.")
print(f"4. The item with the absolute highest sodium content is '{top_sodium.iloc[0]['item']}' from {top_sodium.iloc[0]['restaurant']} with {top_sodium.iloc[0]['sodium']}mg of sodium.")

# --- Unsupervised Learning Analysis ---
print("\n--- Unsupervised Learning Analysis ---")
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA
    from sklearn.ensemble import IsolationForest
    
    # Select numerical features for analysis
    features = ['calories', 'total_fat', 'sat_fat', 'trans_fat', 'cholesterol', 'sodium', 'total_carb', 'fiber', 'sugar', 'protein']
    # Drop rows with missing values in these features to avoid errors
    data_for_ml = df.dropna(subset=features).copy()
    X = data_for_ml[features]
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 1. K-Means Clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    data_for_ml['cluster'] = kmeans.fit_predict(X_scaled)
    
    # 2. PCA for Visualization
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(X_scaled)
    data_for_ml['pca_1'] = principal_components[:, 0]
    data_for_ml['pca_2'] = principal_components[:, 1]
    
    # Plot PCA with Clusters
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=data_for_ml, x='pca_1', y='pca_2', hue='cluster', palette='viridis', style='restaurant', alpha=0.7)
    plt.title('Nutritional Clusters (PCA Visualization)', fontsize=16)
    plt.xlabel('Principal Component 1 (Overall Nutrition Magnitude)', fontsize=12)
    plt.ylabel('Principal Component 2 (Nutritional Composition)', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/nutritional_clusters.png')
    print(f"Saved plot to {output_dir}/nutritional_clusters.png")
    
    # Plot PCA with Restaurants
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=data_for_ml, x='pca_1', y='pca_2', hue='restaurant', alpha=0.6)
    plt.title('Nutritional Landscape by Restaurant (PCA)', fontsize=16)
    plt.xlabel('Principal Component 1', fontsize=12)
    plt.ylabel('Principal Component 2', fontsize=12)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/pca_landscape.png')
    print(f"Saved plot to {output_dir}/pca_landscape.png")

    # 3. Anomaly Detection (Isolation Forest)
    iso_forest = IsolationForest(contamination=0.02, random_state=42)
    data_for_ml['anomaly'] = iso_forest.fit_predict(X_scaled)
    outliers = data_for_ml[data_for_ml['anomaly'] == -1]
    
    print(f"\nDetected {len(outliers)} nutritional outliers.")
    print("Top 5 Outliers:")
    print(outliers[['restaurant', 'item', 'calories', 'sodium']].head(5))

except ImportError:
    print("\n[WARNING] scikit-learn is not installed. Unsupervised learning analysis skipped.")
    print("To run this section, please install scikit-learn: pip install scikit-learn")
