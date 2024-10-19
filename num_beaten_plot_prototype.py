import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Example DataFrame for two enemy sets
data = {
    'Algorithm': ['Baseline EA']*9 + ['NEAT']*9 + ['Baseline EA']*9 + ['NEAT']*9,
    'Enemies Beaten': [0, 1, 2, 3, 4, 5, 6, 7, 8] * 4,
    'Count': [2, 5, 6, 10, 8, 5, 7, 4, 3,   # Baseline EA, Enemy Set 1
              1, 4, 8, 12, 6, 6, 7, 3, 3,   # NEAT, Enemy Set 1
              3, 4, 5, 9, 7, 6, 8, 2, 1,   # Baseline EA, Enemy Set 2
              2, 3, 9, 10, 6, 8, 7, 4, 2]   # NEAT, Enemy Set 2
}

df = pd.DataFrame(data)

# Separate data for each enemy set
df_enemy_set_1 = df.iloc[:18]  # First 18 rows for Enemy Set 1
df_enemy_set_2 = df.iloc[18:]  # Last 18 rows for Enemy Set 2

# Bar width adjustment for touching bars (smaller height to remove gaps)
bar_width = 1.0  # Set to 1 for stacking bars

# Create figure and axes for two side-by-side plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

# Plot for Enemy Set 1 (left plot)
ax1.barh(df_enemy_set_1[df_enemy_set_1['Algorithm'] == 'Baseline EA']['Enemies Beaten'], 
         -df_enemy_set_1[df_enemy_set_1['Algorithm'] == 'Baseline EA']['Count'], 
         height=bar_width, color='tab:blue', edgecolor='black', label='Baseline EA')

ax1.barh(df_enemy_set_1[df_enemy_set_1['Algorithm'] == 'NEAT']['Enemies Beaten'], 
         df_enemy_set_1[df_enemy_set_1['Algorithm'] == 'NEAT']['Count'], 
         height=bar_width, color='tab:orange', edgecolor='black', label='NEAT')

ax1.set_title('Enemy Set 1')
# ax1.set_xlabel('Count of Runs')

# Remove spines (outlines) from the first plot
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['bottom'].set_visible(False)

# Plot for Enemy Set 2 (right plot)
ax2.barh(df_enemy_set_2[df_enemy_set_2['Algorithm'] == 'Baseline EA']['Enemies Beaten'], 
         -df_enemy_set_2[df_enemy_set_2['Algorithm'] == 'Baseline EA']['Count'], 
         height=bar_width, color='tab:blue', edgecolor='black', label='Baseline EA')

ax2.barh(df_enemy_set_2[df_enemy_set_2['Algorithm'] == 'NEAT']['Enemies Beaten'], 
         df_enemy_set_2[df_enemy_set_2['Algorithm'] == 'NEAT']['Count'], 
         height=bar_width, color='tab:orange', edgecolor='black', label='NEAT')

ax2.set_title('Enemy Set 2')
# ax2.set_xlabel('Count of Runs')
ax2.set_xticks([])  # Remove x-axis ticks
ax2.set_xticklabels([])  # Remove x-axis tick labels
ax2.set_yticks([])  # Align y-axis ticks (shared y-axis)
ax2.set_yticklabels([])  # Hide the labels on the right plot's y-axis

# Remove spines (outlines) from the first plot
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax2.spines['bottom'].set_visible(False)

# Add legend only on the first plot
ax2.legend()

ax1.set_ylabel('Number of Enemies Beaten')
ax1.set_xticks([])  # Remove x-axis ticks
ax1.set_xticklabels([])  # Remove x-axis tick labels
ax1.set_yticks(np.arange(9))  # Align y-axis ticks
ax1.set_yticklabels(np.arange(9))

# Display the plots
plt.tight_layout()
plt.show()