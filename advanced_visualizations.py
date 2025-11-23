import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pandas.plotting import parallel_coordinates
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

# Ensure output directories exist
output_dir = 'src/temp'
radar_dir = os.path.join(output_dir, 'radar_charts')
parallel_dir = os.path.join(output_dir, 'parallel_coordinates_charts')

for d in [output_dir, radar_dir, parallel_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

# Select numerical features for analysis
features = ['calories', 'total_fat', 'sat_fat', 'trans_fat', 'cholesterol', 'sodium', 'total_carb', 'fiber', 'sugar', 'protein']

# --- Visualization 1: Radar Chart (Spider Plot) ---
print("\n--- Generating Radar Charts ---")

# Calculate average nutritional profile per restaurant
avg_nutrition = df.groupby('restaurant')[features].mean()

# Get max values for scaling context
max_values = df[features].max()
max_values_str = "\n".join([f"{f}: {max_values[f]}" for f in features])

# Normalize data to 0-1 range
scaler = MinMaxScaler()
# We fit on the WHOLE dataset to ensure the scale is global (0 = min in dataset, 1 = max in dataset)
# But here we are plotting averages. It's better to scale based on the max possible values in the dataset
# so that "1.0" means "This average is as high as the highest item in the dataset" (which is unlikely)
# OR we can scale based on the range of the averages themselves.
# Let's scale based on the max values of the *averages* to see relative differences between chains clearly.
# actually, user asked for "how far it's going in each dimension".
# If we scale 0-1 based on the dataset max, the averages will look very small (because no chain averages 6000mg sodium).
# Let's scale based on the max of the AVERAGES, but print the max of the AVERAGES in the legend.
scaler_avg = MinMaxScaler()
avg_nutrition_scaled = pd.DataFrame(scaler_avg.fit_transform(avg_nutrition), columns=avg_nutrition.columns, index=avg_nutrition.index)

# But wait, if we want to see "how far it goes", maybe we should use the dataset max to show "how unhealthy is this average compared to the worst item"?
# Let's stick to scaling the averages against each other (0-1 range of averages) so the shapes are distinct,
# and provide the "Max Average Value" in the legend.
max_avg_values = avg_nutrition.max()
legend_str = "Max Average Values (Scale 1.0):\n" + "\n".join([f"{f}: {max_avg_values[f]:.1f}" for f in features])

def make_radar_chart(data, title, filename, legend_text=None, specific_restaurant=None):
    categories = list(data.columns)
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    plt.figure(figsize=(12, 10))
    ax = plt.subplot(111, polar=True)
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75], ["25%", "50%", "75%"], color="grey", size=7)
    plt.ylim(0, 1)

    palette = sns.color_palette("tab10", len(data))
    
    # If specific_restaurant is set, only plot that one
    if specific_restaurant:
        row = data.loc[specific_restaurant]
        values = row.values.flatten().tolist()
        values += values[:1]
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=specific_restaurant, color='blue')
        ax.fill(angles, values, color='blue', alpha=0.2)
    else:
        # Plot all
        for i, (name, row) in enumerate(data.iterrows()):
            values = row.values.flatten().tolist()
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=name, color=palette[i])
            ax.fill(angles, values, color=palette[i], alpha=0.05)

    plt.title(title, size=20, y=1.1)
    
    # Add legend for scale
    if legend_text:
        plt.figtext(0.85, 0.5, legend_text, fontsize=9, bbox=dict(facecolor='white', alpha=0.8))

    if not specific_restaurant:
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved radar chart to {filename}")

# 1. All Restaurants Radar
make_radar_chart(avg_nutrition_scaled, "Average Nutritional Profile (All Chains)", f'{output_dir}/radar_chart.png', legend_str)

# 2. Individual Radar Charts
for restaurant in avg_nutrition_scaled.index:
    filename = os.path.join(radar_dir, f"{restaurant.replace(' ', '_')}_radar.png")
    make_radar_chart(avg_nutrition_scaled, f"Nutritional Profile: {restaurant}", filename, legend_str, specific_restaurant=restaurant)


# --- Visualization 2: Parallel Coordinates ---
print("\n--- Generating Parallel Coordinates Plots ---")

# Normalize the entire dataset for parallel coordinates (using global min/max)
scaler_global = MinMaxScaler()
df_normalized = df.copy()
df_normalized[features] = scaler_global.fit_transform(df[features])

# 1. All Restaurants Parallel Coordinates
plt.figure(figsize=(15, 8))
parallel_coordinates(df_normalized[['restaurant'] + features], 'restaurant', colormap=plt.get_cmap("tab10"), alpha=0.3)
plt.title('Parallel Coordinates: Nutritional Flow (All Chains)', fontsize=16)
plt.xlabel('Nutritional Features', fontsize=12)
plt.ylabel('Normalized Value (0-1)', fontsize=12)
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f'{output_dir}/parallel_coordinates.png')
plt.close()
print(f"Saved parallel coordinates plot to {output_dir}/parallel_coordinates.png")

# 2. Individual Parallel Coordinates
for restaurant in df['restaurant'].unique():
    plt.figure(figsize=(15, 8))
    
    # Plot all data in grey background for context
    parallel_coordinates(df_normalized[['restaurant'] + features], 'restaurant', color='lightgrey', alpha=0.1)
    
    # Highlight specific restaurant
    subset = df_normalized[df_normalized['restaurant'] == restaurant]
    # We need to hack parallel_coordinates a bit to show just one color, or just plot manually.
    # Easiest is to just plot the subset on top.
    # But parallel_coordinates expects a class column.
    
    # Let's just plot the subset using parallel_coordinates but with a specific color
    # We create a temporary dataframe where the 'restaurant' column is just the name
    # It will automatically assign a color, but since there's only one class, it's fine.
    parallel_coordinates(subset[['restaurant'] + features], 'restaurant', color='blue', alpha=0.5)
    
    plt.title(f'Parallel Coordinates: {restaurant}', fontsize=16)
    plt.xlabel('Nutritional Features', fontsize=12)
    plt.ylabel('Normalized Value (0-1)', fontsize=12)
    plt.xticks(rotation=45)
    # Remove legend as it's redundant (or just shows the one restaurant)
    plt.legend().remove()
    plt.tight_layout()
    
    filename = os.path.join(parallel_dir, f"{restaurant.replace(' ', '_')}_parallel.png")
    plt.savefig(filename)
    plt.close()
    print(f"Saved parallel coordinates to {filename}")

# 3. Average Parallel Coordinates (Combined)
print("\n--- Generating Combined Average Parallel Coordinates Plot ---")

# We already have avg_nutrition, but we need to normalize it globally (0-1) for the plot
# We can reuse the scaler_global we fit on the whole dataset, or fit on the averages.
# Fitting on the averages makes the differences between chains more distinct.
# Fitting on the global dataset shows how the averages compare to the "worst possible" items.
# Let's fit on the averages to maximize contrast between chains.
scaler_avg_pc = MinMaxScaler()
avg_nutrition_pc = avg_nutrition.copy().reset_index()
avg_nutrition_pc[features] = scaler_avg_pc.fit_transform(avg_nutrition_pc[features])

plt.figure(figsize=(15, 8))
parallel_coordinates(avg_nutrition_pc, 'restaurant', colormap=plt.get_cmap("tab10"), alpha=0.8, linewidth=3)
plt.title('Average Nutritional Profile by Restaurant (Parallel Coordinates)', fontsize=16)
plt.xlabel('Nutritional Features', fontsize=12)
plt.ylabel('Relative Value (0=Lowest Avg, 1=Highest Avg)', fontsize=12)
plt.xticks(rotation=45)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.savefig(f'{output_dir}/average_parallel_coordinates.png')
plt.close()
print(f"Saved average parallel coordinates plot to {output_dir}/average_parallel_coordinates.png")
