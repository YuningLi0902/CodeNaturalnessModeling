import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

# Data
models = [
    "CodeT5-base", "BERT-large", "BERT-base", "CodeLlama-13B",
    "CodeT5-large", "UniXcoder", "InCoder-1B", "InCoder-6B", "CodeBERT"
]

rank_sums = [9.125, 21.25, 24.25, 26.125, 29.875, 33.375, 36.25, 45.625, 51.75]

# Sort the data by rank sum for better visualization
sorted_indices = np.argsort(rank_sums)
sorted_models = np.array(models)[sorted_indices][::-1]
sorted_rank_sums = np.array(rank_sums)[sorted_indices][::-1]

# Create a list of colors based on the rank sums
colors = ['#08306B', '#0b559F', '#71B1D7', '#549FCD', '#71B1D7', '#94C4DF', '#A5CDE3', '#BCD7EB', '#CCE5FF']

# Create horizontal bar chart
plt.figure(figsize=(12, 6))
bars = plt.barh(sorted_models, sorted_rank_sums, color=colors)  # '#b3b1a4'

# Adding labels
plt.xlabel('Rank Sum Value')

# Adding data labels on the bars
for bar in bars:
    width = bar.get_width()
    plt.text(width, bar.get_y() + bar.get_height() / 2,
             f'{width:.3f}',
             ha='left', va='center')

# Display the plot
plt.tight_layout()
plt.show()
plt.savefig('rank_sum.png')
