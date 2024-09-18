import matplotlib
import matplotlib.pyplot as plt

matplotlib.use('Agg')

# Data for the two groups
c_common = [0.1952, 0.3308, 0.3227, 0.228, 0.3056, 0.3041, 0.2881, 0.2851, 0.3154, 0.3466,
            0.3671, 0.362, 0.2096, 0.242, 0.2369, 0.0802, 0.1841, 0.212, 0.1592, 0.176,
            0.2179, 0.2536, 0.2163, 0.1844, 0.2294, 0.1679, 0.1403, 0.2413, 0.2401, 0.1882,
            0.147, 0.2556, 0.06, 0.0208, 0.0797, 0.0957, 0.0341, 0.0037, 0.019, 0.2475,
            0.2132, 0.19, 0.1701, 0.1641, 0.0765, 0.0631, 0.0414, 0.0508, 0.0986, 0.1121,
            0.1253, 0.1099, 0.0753, 0.1098, 0.0721, 0.0754, 0.097, 0.0772, 0.0508, 0.0455]
c_fix = [0.015, 0.0067, 0.0325, 0.1846, 0.1174, 0.0825, 0.0898, 0.0909, 0.0435, 0.0091,
         0.0109, 0.0038, 0.0352, 0.0317, 0.0487, 0.0585, 1.49E-04, 0.0027, 0.0066, 0.008,
         0.0789, 0.1101, 0.1253, 0.114, 0.1614, 0.0986, 0.0816, 0.0402, 0.1668, 0.1143,
         0.0804, 0.0343, 0.0752, 0.0018, 0.0585, 0.0604, 0.005, 0.0228, 0.049, 0.1062,
         0.1837, 0.1794, 0.1588, 0.1484, 0.0948, 0.079, 0.0787, 0.0952, 0.0481, 0.0594,
         0.082, 0.0892, 0.0094, 0.0231, 0.0274, 0.0402, 0.0758, 0.1076, 0.1744, 0.1853]

# Indices for the x-axis
indices = list(range(1, len(c_common) + 1))

# Create a scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(indices, c_common, color='blue', label="Cohen's d-common")
plt.scatter(indices, c_fix, color='red', label="Cohen's d-fixed")

# Adding labels and title
plt.xlabel('Index')
plt.ylabel("Cohen's d")
plt.legend(loc='upper right')

# Set the x-axis limits to ensure it starts from 1 and ends at 52
plt.xlim(0, len(c_common) + 1)

# Adding vertical dashed lines at every multiple of 4
for x in range(4, len(c_common)+1, 4):
    plt.axvline(x=x, color='gray', linestyle='--', alpha=0.7)
# Adding a horizontal dashed line at y=0.20
plt.axhline(y=0.20, color='gray', linestyle='--', alpha=0.7)

plt.savefig('cohens_d.png')
