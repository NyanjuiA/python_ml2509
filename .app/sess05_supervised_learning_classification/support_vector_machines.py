"""
================================================================================
Python script to demonstrate Support Vector Machines (SVMs)
================================================================================
The script uses SVMs to classify a fruit as either an apple or orange based
on its weight and size


Requirements
--------------------------
 pip install matplotlib numpy scikit-learn

Author : Nyanjui
Date   : 05 September 2026
"""
# ================================================================================
# SECTION 0: Import the required modules
# ================================================================================
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ================================================================================
# SECTION 1: Dataset generation and preparation
# ================================================================================
# Synthetic dataset for classifying fruits (apples & oranges) using
# 'weight' and 'size' as features
np.random.seed(42) # For reproducibility

# Generate the data for apples
weight_apples = np.random.normal(150, 18,50) # Weighs around 150 grams
size_apples = np.random.normal(7, 1.6,50) # Size about 7 cm
label_apples = np.zeros(50) # Use label zero '0' for apples

# Generate the data for oranges
weight_oranges = np.random.normal(200, 18,50) # Weighs around 200 grams
size_oranges = np.random.normal(8.5, 1.6,50) # Size about 8.5 cm
label_oranges = np.ones(50) # Use label one '1' for oranges

# Combine the above data into a single dataset
weight = np.concatenate((weight_apples, weight_oranges))
size = np.concatenate((size_apples, size_oranges))
labels = np.concatenate((label_apples, label_oranges))

# Feature matrix and target vector
X = np.column_stack((weight, size))
y = labels

# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ================================================================================
# SECTION 2: Model initialisation, training, prediction, and evaluation
# ================================================================================
# Initialise the SVM model with a linear kernel
svm_clf = SVC(kernel='linear', C = 1) # .fit(X_train, y_train)

# Train the model
svm_clf.fit(X_train, y_train)

# Make predictions
y_pred = svm_clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['apples', 'oranges'])

# ================================================================================
# SECTION 3: Display and visualise the results
# ================================================================================
# Display the results
print(f"Accuracy of the SVM model: {accuracy}")
print(f"Confusion matrix of the SVM model: \n{conf_matrix}")
print(f"Classification report of the SVM model: \n{class_report}")

# Function to visualise the decision boundary
def plot_decision_boundaries(X, y, model):
   plt.figure(figsize=(10, 8))

   # plot the decision boundary
   ax = plt.gca()
   xlim = ax.get_xlim()  # Limit of the x-axis
   ylim = ax.get_ylim()  # Limit of the y-axis

   # Create a mesh to plot the decision boundary
   xx, yy = np.meshgrid(  # Generate 100 evenly spaced point btwn. start and end
      np.linspace(xlim[0], xlim[1], 100),
      np.linspace(ylim[0], ylim[1], 100))
   Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]) # Flatten the 2D array into 1d
   Z = Z.reshape(xx.shape)

   # Plot the decision boundary and margins
   plt.contourf(xx, yy, Z > 0, alpha = 0.2, colors=["#ffaaaa", '#aaaaff'])
   plt.contour(xx, yy, Z, colors='k', levels=[-1,0,1], linestyles=['--', '-', '--'])

   # Plot the support vectors
   plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
               facecolors='none', edgecolors='k', label='Support Vectors')

   # Plot the data points
   plt.scatter(X[:,0],X[:,1], s=50, c=y, cmap=plt.cm.Paired, edgecolors='k')
   plt.xlabel('Weight (grams)')
   plt.ylabel('Size (cm)')
   plt.title('SVM Decision Boundary and Support Vectors')
   plt.legend()
   plt.show()  # End of visualisation function

# Visualise the decision boundary and support vectors by calling the 'plot_decision_boundary()' function
plot_decision_boundaries(X, y, svm_clf)