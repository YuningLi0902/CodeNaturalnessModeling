import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

# Data for the two groups
c_common = [0.2, 0.33, 0.32, 0.23, 0.31, 0.3, 0.29, 0.29, 0.32, 0.35,
            0.37, 0.36, 0.21, 0.24, 0.24, 0.08, 0.18, 0.21, 0.16, 0.18,
            0.23, 0.17, 0.14, 0.24, 0.24, 0.19, 0.15, 0.26, 0.06, 0.02,
            0.08, 0.1, 0.03, 3.70E-03, 0.02, 0.25, 0.21, 0.19, 0.17,
            0.16, 0.08, 0.06, 0.04, 0.05, 0.1, 0.11, 0.13, 0.11, 0.08,
            0.11, 0.07, 0.08]
c_fix = [0.01, 0.01, 0.03, 0.18, 0.12, 0.08, 0.09, 0.09, 0.04,
         0.01, 0.01, 3.77E-03, 0.04, 0.03, 0.05, 0.06, 1.49E-04,
         2.72E-03, 0.01, 0.01, 0.16, 0.1, 0.08, 0.04, 0.17,
         0.11, 0.08, 0.03, 0.08, 1.79E-03, 0.06, 0.06, 0.01,
         0.02, 0.05, 0.11, 0.18, 0.18, 0.16, 0.15, 0.09,
         0.08, 0.08, 0.1, 0.05, 0.06, 0.08, 0.09, 0.01,
         0.02, 0.03, 0.04]

# Indices for the x-axis
indices = list(range(1, len(c_common) + 1))

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(indices, c_common, color='blue', label="Cohen's d-common")
plt.scatter(indices, c_fix, color='red', label="Cohen's d-fix")

# Adding labels and title
plt.xlabel('Index')
plt.ylabel("Cohen's d")
plt.legend(loc='upper right')

# Set the x-axis limits to ensure it starts from 1 and ends at 52
plt.xlim(0, len(c_common) + 1)

# Adding vertical dashed lines at every multiple of 4
for x in range(4, 53, 4):
    plt.axvline(x=x, color='gray', linestyle='--', alpha=0.7)
# Adding a horizontal dashed line at y=0.20
plt.axhline(y=0.20, color='gray', linestyle='--', alpha=0.7)

plt.savefig('cohens_d.png')
