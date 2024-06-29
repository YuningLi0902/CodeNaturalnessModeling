import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

# Data
models = [
    "CodeT5-base", "BERT-large", "BERT-base", "CodeT5-large",
    "UniXcoder", "InCoder-1b", "InCoder-6b", "CodeBERT"
]

rank_sums = [8.875, 17.5, 20.75, 25.375, 28.5, 30.375, 38, 44]

# Sort the data by rank sum for better visualization
sorted_indices = np.argsort(rank_sums)
sorted_models = np.array(models)[sorted_indices][::-1]
sorted_rank_sums = np.array(rank_sums)[sorted_indices][::-1]

# Create horizontal bar chart
plt.figure(figsize=(12, 6))
bars = plt.barh(sorted_models, sorted_rank_sums, color='#b3b1a4')

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
