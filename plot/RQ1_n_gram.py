import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

# Data from the table
models = [
    "BERT-base", "BERT-large", "CodeBERT", "UniXcoder",
    "CodeT5-base", "CodeT5-large", "InCoder-1B", "InCoder-6B", "CodeLlama-13B"
]

normal_code = [7.69, 9.69, 4.49, 9.18, 8.76, 5.84, 5.95, 5.93, 4.47]
buggy_code = [8.56, 10.85, 4.66, 9.43, 9.95, 6.23, 6.12, 6.06, 4.66]
fixed_code = [7.85, 9.92, 4.51, 9.21, 9.10, 5.80, 5.94, 6.04, 4.52]

# Setting the positions and width for the bars
pos = np.arange(len(models))
width = 0.25

# Plotting the bars
fig, ax = plt.subplots(figsize=(12, 6))

bar1 = ax.bar(pos - width, normal_code, width, label='Common Code', color='#42a1ff')
bar2 = ax.bar(pos, buggy_code, width, label='Buggy Code', color='#d64a4a')
bar3 = ax.bar(pos + width, fixed_code, width, label='Fixed Code', color='#3cc32c')

# Adding labels and title
ax.set_ylabel('Code Naturalness', fontsize=12)
ax.set_xticks(pos)
ax.set_xticklabels(models)
ax.legend()

ax.set_ylim(4, max(max(normal_code), max(buggy_code), max(fixed_code)) + 1)


# Adding data labels on top of each bar
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


add_labels(bar1)
add_labels(bar2)
add_labels(bar3)

# Save the plot as a PNG file
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('code_naturalness.png')
