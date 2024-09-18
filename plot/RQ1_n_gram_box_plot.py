import matplotlib
import matplotlib.pyplot as plt
import pickle

matplotlib.use('Agg')

# Data from the table
models = [
    "BERT-base", "BERT-large", "CodeBERT", "UniXcoder",
    "CodeT5-base", "CodeT5-large", "InCoder-1B", "InCoder-6B", "CodeLlama-13B"
]

data_file = 'n_gram_model_data.pickle'
with open(data_file, 'rb') as f:
    data_dict = pickle.load(f)
    f.close()
data_list = list(data_dict.values())

colors = ['#ccdeff', '#ffe5e5', '#99ff99']
label_size = 14
title_size = 16
plt.figure()
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
for i in range(3):
    for j in range(3):
        ax = axs[i, j]
        ax.set_xticklabels(['common', 'buggy', 'fixed'], fontsize=label_size)
        ax.tick_params(axis='y', labelsize=label_size-1)
        result = ax.boxplot(data_list[3 * i + j], patch_artist=True, showmeans=True, meanline=True,
                            medianprops={'color': 'black', 'linewidth': 2},
                            meanprops={'color': 'red', 'ls': '-', 'lw': 2})
        for patch, color in zip(result['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_title(models[3 * i + j], fontsize=title_size, fontweight='bold')
plt.tight_layout()
plt.show()
plt.savefig('boxplot.png')

plt.figure()
fig, axs = plt.subplots(3, 3, figsize=(15, 15))
for i in range(3):
    for j in range(3):
        ax = axs[i, j]
        ax.set_xticklabels(['common', 'buggy', 'fixed'], fontsize=label_size)
        ax.tick_params(axis='y', labelsize=label_size-1)
        result = ax.boxplot(data_list[3 * i + j], patch_artist=True, showmeans=True, showfliers=False, meanline=True,
                            medianprops={'color': 'black', 'linewidth': 2},
                            meanprops={'color': 'red', 'ls': '-', 'lw': 2}, whis=0)
        for patch, color in zip(result['boxes'], colors):
            patch.set_facecolor(color)
        ax.set_title(models[3 * i + j], fontsize=title_size, fontweight='bold')
plt.tight_layout()
plt.show()
plt.savefig('boxplot_zoom_in.png')
