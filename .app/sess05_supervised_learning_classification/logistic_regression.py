"""
================================================================================
Python script to demostrate Logistic Regression Classification to determine
whether a person buys a product or not
================================================================================


Requirements
--------------------------
 pip install matplotlib numpy pandas scikit-learn

Author : Nyanjui
Date   : 04 September 2026
"""
# ================================================================================
# SECTION 0: Import the required modules
# ================================================================================
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ================================================================================
# SECTION 1: Create and display a sample dataset
# ================================================================================
# Features: [ Age, Estimated Salary]
# Labels: [ 0 -> not buy, 1 -> buy ]
data = {
   'Age': [22, 25, 47, 52, 46, 56, 55, 60, 62, 61, 30, 32, 34, 35, 40],
   'Estimated Salary': [15000, 29000, 35000, 43000, 70000, 80000, 20000, 25000, 90000, 110000, 60000, 52000, 78000,
                        65000, 120000],
   'Buy': [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1]
}

# Convert the above dictionary into a Pandas DataFrame
df = pd.DataFrame(data)

# Display the dataset
print(f"Training Dataset:\n{df}")

# ================================================================================
# SECTION 2: Feature seperation, data split and feature scale
# ================================================================================
# Seperate the features and the target
X = df[['Age', 'Estimated Salary']]
y = df['Buy']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale the features for better model performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================================================================================
# SECTION 3: Logistic Regression model initialisation, training, prediction & evaluation
# ================================================================================
# Initialise the Logistic Regression Model
log_reg = LogisticRegression()

# Train the model
log_reg.fit(X_train, y_train)

# Make predictions
y_pred = log_reg.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['Not Buy', 'Buy'])

# ================================================================================
# SECTION 4: Display results and visualise data
# ================================================================================
# Display the results
print(f"Accuracy of the Logistic Regression model: {accuracy}")
print(f"\nConfusion Matrix of the Logistic Regression model:\n{conf_matrix}")
print(f"\nClassification report of the Logistic Regression model:\n{class_report}")

# Visualise the data
# Decision boundary (for 2D)
# Create a mesh grid for plotting the decision boundary
x_min, x_max = X_train[:,0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:,1].min() - 1, X_train[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

# Predict on mesh grid
Z = log_reg.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary and data points
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.2,cmap=plt.cm.Paired)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', s=50, edgecolors='w',
            cmap=plt.cm.Paired, label='Training Set')
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='x', s=50,
            cmap=plt.cm.Paired, label='Test Set')
plt.xlabel('Age (scale)')
plt.ylabel('Estimated Salary (scale)')
plt.title('Logistic Regression Decision Boundary')
plt.legend()
plt.show()

