from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
#2. Finite State Machine Generalization:
'''(a) Implement a program that automatically creates a set of ifthen clauses from the training table of a binary dataset of your choice. Implement different strategies to minimize the
#number of if-then clauses. Document your strategies, the number of resulting conditional clauses, and the accuracy achieved. '''

'''IRIS-Dataset: This data sets consists of 3 different types of irises’ (Setosa, Versicolour, and Virginica) petal and sepal length, stored in a 150x4 numpy.ndarray
The rows being the samples and the columns being: Sepal Length, Sepal Width, Petal Length and Petal Width. We will be converting the iris type to Setosa vs Non Setosa for binary purposes'''
iris = datasets.load_iris()
X = iris.data[:, :]
y = iris.target
y = [0 if label == 1 else 1 for label in y] 

# 0 represents Versicolour / 1 represents not Versicolour

# ****************
# ** STRATEGY 1 **
# ****************

'''Basic Memorization: if-then clause for every unique data point '''

def strategy1(input, output, n):
  '''
  Memorization: creates if-then clause for every instance of the data
  Accuracy of 100 %, represents MEC with the upper bound of if-then clauses
  '''
  if input[n]:
    return output[n]


# ****************
# ** STRATEGY 2 **
# ****************

'''
Use sklearn's decision tree classifier to create a decision tree and prune it to lessen the amount of if then clauses
we see that one feature is deterministic of the output, so we end up with one if-then clause to get 100% accuracy
'''

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=42)

#remove random splitter to decrease leaf nodes
clf = DecisionTreeClassifier(random_state=42, splitter="random")
clf.fit(X_train, y_train)

def get_clauses(tree, feature_names, node_idx=0, clauses=None):
  print(clauses)
  if clauses is None:
    clauses = []

  node = tree.tree_.feature[node_idx]
  class_label = tree.classes_[np.argmax(tree.tree_.value[node_idx][0])]

  if node == -2:
    clause = 'then class = {}'.format(class_label)
    clauses.append(' '.join(clause.split()))
    return clauses

  feature_name = feature_names[node]
  threshold = tree.tree_.threshold[node_idx]
  clause = "If {} <= {}".format(feature_name, threshold)
  clauses.append(' '.join(clause.split()))

  left_idx = tree.tree_.children_left[node_idx]
  right_idx = tree.tree_.children_right[node_idx]
  
  clauses = get_clauses(tree, feature_names, left_idx, clauses)
  clauses = get_clauses(tree, feature_names, right_idx, clauses)
  

  return clauses


feature_names = iris.feature_names[:]
print(feature_names)
print("Test value: ", clf.tree_.value)
clauses = get_clauses(clf, feature_names)

print('If-Then Clauses:')

for clause in clauses:
  print(clause)

print(export_text(clf, feature_names=iris.feature_names[:]))

accuracy = clf.score(X_test, y_test)
print(f'\nAccuracy: {accuracy:.2f}')
print(f'Number of If-Then Clauses: {len(clauses)}')

'''(b) Use the algorithms developed in (a) on different datasets. Again, observe how your choices make a difference. '''

'''banana_quality dataset: col_names = Size, Weight, Sweetness, Softness, HarvestTime, Ripeness, Acidity, Quality'''
banana_df = pd.read_csv('banana_quality.csv')
banana_X, banana_y = banana_df.iloc[:,0:7].to_numpy(), banana_df.iloc[:,7].apply(lambda x: 1 if x == 'Good' else 0).to_numpy()

# ****************
# ** STRATEGY 1 **
# ****************


# ****************
# ** STRATEGY 2 **
# ****************

'''(c) Finally, use the programs developed in (a) on a completely random dataset, generated artificially. Vary your strategies but also the number of input columns as well as the number
of instances. How many if-then clauses do you need? '''


