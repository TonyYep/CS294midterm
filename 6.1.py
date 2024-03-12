# 1) Logic Definition of Generalization:
#(a) Show empirically that the information limit of 2 prediction bits per parameter also holds for nearest neighbors.

import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

def generate_dataset(N, D, noise_std = 0.1):
  X = np.random.uniform(0, 1, size=(N, D))
  y = np.sum(X, axis=1) + np.random.normal(0, noise_std, size=N)

  return X, y

def max_distinct_predictions(X, y):
  neighbor = NearestNeighbors(n_neighbors=2)
  neighbor.fit(X)
  distances, indices = neighbor.kneighbors(X, return_distance=True)
  predictions = y[indices[:, 0]]

  return len(np.unique(predictions))

N_VAL = [100, 500, 1000, 5000, 10000]
D_VAL = [1, 2, 5, 10, 20]


res = []

for N in N_VAL:
  for D in D_VAL:
    
    X, y = generate_dataset(N, D)
    max_preds = max_distinct_predictions(X, y)
    num_params = N * D
    res.append((num_params, max_preds))

    

res = np.array(res)

#scatter plot for regression
tempX, tempy = generate_dataset(50, 10)
tempMax = max_distinct_predictions(50, 10)

print("Max: ", tempMax)


"""
plt.figure(figsize=(8, 6))
plt.loglog(res[:, 0], res[:, 1], 'o')
plt.xlabel('Number of Params')
plt.ylabel('Max Distinct Predictions')
plt.title('Information Limit for Nearest Neighbor Regression')

coeffs = np.polyfit(np.log(res[:, 0]), np.log(res[:, 1]), 1)

slope = coeffs[0]

print(f'Slope of line: {slope:.2f}')

x_fit = np.logspace(np.log10(min(res[:, 0])), np.log10(max(res[:, 0])), 100)
y_fit = np.exp(coeffs[1] + coeffs[0] * np.log(x_fit))
plt.plot(x_fit, y_fit, 'r--', label=f'Slope = {slope:.2f}')
plt.legend()
plt.show()"""

'''
From the figure we can observe a linear relationship between the number of parameters and number of max distinct predictions. This empiricsupports the information limit of 2 prediction bits
per parameter.
'''

#(b) Extend your experiments to multi-class classification.


