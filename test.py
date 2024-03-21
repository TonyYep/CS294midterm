from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate a synthetic multi-class dataset with 4 classes
X, y = make_blobs(n_samples=1000, centers=4, random_state=0)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Define a list of k values to experiment with
k_values = range(1, 200)  # Experiment with k from 1 to 20

# List to store accuracy for each k
accuracy_scores = []

# Loop through different k values
for k in k_values:
    # Create a KNN classifier with current k
    knn = KNeighborsClassifier(n_neighbors=k)  # Consider k as a complexity measure

    # Train the model
    knn.fit(X_train, y_train)

    # Make predictions on test data
    y_pred = knn.predict(X_test)

    # Calculate accuracy
    accuracy = knn.score(X_test, y_test)
    accuracy_scores.append(accuracy)

# Plot accuracy vs k
plt.plot(k_values, accuracy_scores)
plt.xlabel("Number of Neighbors (k)")
plt.ylabel("Accuracy")
title = "Accuracy vs. Model Complexity (k-NN) - Synthetic 4-Class"
plt.title(title)
plt.show()