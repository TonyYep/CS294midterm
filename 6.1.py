# 1) Logic Definition of Generalization:
#(a) Show empirically that the information limit of 2 prediction bits per parameter also holds for nearest neighbors.

from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import tree
import pandas as pd
import math
import random

np.random.seed(999)

def generate_dataset(n, d):
  data = []
  for i in range(n):
    val = np.random.randint(0 , 100 , d)
    data.append(val)

  classes = np.random.randint(0, 2, n)

  return data, classes

def find_accuracy(data, classes, test_data, test_classes):
  neighbor = KNeighborsClassifier(n_neighbors = 1)
  neighbor.fit(data, classes)
  predictions = neighbor.predict(test_data)
  
  return sum(predictions == test_classes) / len(test_classes)

def counting_formula(n, d):
  if n == 1 or d == 1:
    return 2
  if n == 0 or d == 0:
    return 0
  else:
    values = 0
    for i in range(d):
      values += math.comb(n-1, i)
    return 2 * values

def random_remove(remove_count, remove_data, remove_classes):
  new_ind = random.sample(range(len(remove_data)), len(remove_data) - remove_count)
  new_data = []
  new_class = []
  for x in new_ind:
    new_data.append(remove_data[x])
    new_class.append(remove_classes[x])
  return new_data, new_class

def find_memorization(n, d):
  data, classes = generate_dataset(n, d)
  counts = []
  for x in range(2 ** d):
    instance_count = 0
    memorization = 1
    temp_data = data
    temp_classes = classes
    removed_count = 0
    while memorization == 1:
      memorization = find_accuracy(temp_data, temp_classes, data, classes)
      instance_count = len(temp_data)
      temp_data, temp_classes = random_remove(removed_count, temp_data, temp_classes)
      removed_count += 1
    counts.append(instance_count)

  return counts

#(b) Extend your experiments to multi-class classification.


