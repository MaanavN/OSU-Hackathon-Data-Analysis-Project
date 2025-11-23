import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy import stats

# Set style
sns.set_theme(style="whitegrid")

# Ensure output directory exists
output_dir = 'src/temp'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

print("--- Loading NHANES Data ---")
# Load datasets
try:
    diet_df = pd.read_csv('src/temp/nhanes_diet.csv')
    quest_df = pd.read_csv('src/temp/nhanes_questionnaire.csv')
    print(f"Loaded Diet Data: {diet_df.shape}")
    print(f"Loaded Questionnaire Data: {quest_df.shape}")
except FileNotFoundError:
    print("Error: NHANES CSV files not found. Please ensure they are downloaded.")
    exit(1)

# Merge on SEQN (Sequence Number - Participant ID)
print("--- Merging Datasets ---")
merged_df = pd.merge(diet_df, quest_df, on='SEQN', how='inner')
print(f"Merged Data Shape: {merged_df.shape}")

# Define variables of interest
# Diet:
# DR1TSFAT: Total saturated fatty acids (gm)
# DR1TSODI: Sodium (mg)
# DR1TSUGR: Total sugars (gm)

# Questionnaire:
# DIQ010: Doctor told you have diabetes? (1=Yes, 2=No, 3=Borderline)

variables = {
    'DR1TSFAT': 'Saturated Fat (g)',
    'DR1TSODI': 'Sodium (mg)',
    'DR1TSUGR': 'Sugar (g)',
    'DIQ010': 'Diabetes Status'
}

# Filter for these columns
analysis_df = merged_df[list(variables.keys())].copy()

# Rename columns for readability
analysis_df.rename(columns=variables, inplace=True)

# Data Cleaning
print("--- Cleaning Data ---")
# Remove rows with missing values
initial_count = len(analysis_df)
analysis_df.dropna(inplace=True)
print(f"Removed {initial_count - len(analysis_df)} rows with missing values.")

# Filter for clear Yes/No diabetes status (1=Yes, 2=No), exclude borderline (3)
analysis_df = analysis_df[analysis_df['Diabetes Status'].isin([1, 2])]
print(f"Filtered for clear diabetes status (Yes/No): {len(analysis_df)} participants")

# Recode: 1=Diabetic, 0=Non-Diabetic
analysis_df['Diabetic'] = (analysis_df['Diabetes Status'] == 1).astype(int)
analysis_df['Diabetes Group'] = analysis_df['Diabetic'].map({1: 'Diabetic', 0: 'Non-Diabetic'})

# Remove unrealistic dietary values (e.g., 0 intake)
analysis_df = analysis_df[(analysis_df['Saturated Fat (g)'] > 0) & (analysis_df['Sodium (mg)'] > 0)]
print(f"Final Data Count for Analysis: {len(analysis_df)}")

diabetic_count = analysis_df['Diabetic'].sum()
non_diabetic_count = len(analysis_df) - diabetic_count
print(f"  - Diabetic: {diabetic_count}")
print(f"  - Non-Diabetic: {non_diabetic_count}")

# --- Analysis 1: Box Plots Comparison ---
print("\n--- Generating Box Plots ---")

nutrients = ['Saturated Fat (g)', 'Sodium (mg)', 'Sugar (g)']
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, nutrient in enumerate(nutrients):
    sns.boxplot(data=analysis_df, x='Diabetes Group', y=nutrient, ax=axes[i], palette='Set2')
    axes[i].set_title(f'{nutrient} by Diabetes Status', fontsize=14)
    axes[i].set_xlabel('')
    axes[i].set_ylabel(nutrient, fontsize=12)

plt.tight_layout()
plt.savefig(f'{output_dir}/diabetes_boxplots.png')
print(f"Saved box plots to {output_dir}/diabetes_boxplots.png")

# --- Analysis 2: Mean Comparison Bar Chart ---
print("\n--- Generating Mean Comparison Chart ---")

mean_comparison = analysis_df.groupby('Diabetes Group')[nutrients].mean().T
mean_comparison.plot(kind='bar', figsize=(10, 6), color=['#66c2a5', '#fc8d62'], width=0.7)
plt.title('Average Dietary Intake: Diabetic vs Non-Diabetic', fontsize=16)
plt.xlabel('Nutrient', fontsize=12)
plt.ylabel('Average Daily Intake', fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='Group')
plt.tight_layout()
plt.savefig(f'{output_dir}/diabetes_mean_comparison.png')
print(f"Saved mean comparison to {output_dir}/diabetes_mean_comparison.png")

# --- Statistical Testing ---
print("\n--- Statistical Testing (T-Tests) ---")
print("Comparing Diabetic vs Non-Diabetic groups:\n")

for nutrient in nutrients:
    diabetic_data = analysis_df[analysis_df['Diabetic'] == 1][nutrient]
    non_diabetic_data = analysis_df[analysis_df['Diabetic'] == 0][nutrient]
    
    t_stat, p_value = stats.ttest_ind(diabetic_data, non_diabetic_data)
    
    mean_diabetic = diabetic_data.mean()
    mean_non_diabetic = non_diabetic_data.mean()
    percent_diff = ((mean_diabetic - mean_non_diabetic) / mean_non_diabetic) * 100
    
    print(f"{nutrient}:")
    print(f"  Diabetic Mean: {mean_diabetic:.2f}")
    print(f"  Non-Diabetic Mean: {mean_non_diabetic:.2f}")
    print(f"  Difference: {percent_diff:+.1f}%")
    print(f"  T-statistic: {t_stat:.3f}")
    print(f"  P-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print(f"  *** SIGNIFICANT (p < 0.05) ***")
    else:
        print(f"  Not significant (p >= 0.05)")
    print()
