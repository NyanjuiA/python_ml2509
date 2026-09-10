"""
================================================================================
Python script to demonstrate Neural Networks
================================================================================
This script demonstrates Neural Networks to classify a fruit as either an apple
or an orange based on its weight and colour

Requirements
--------------------------
 pip install matplotlib numpy scikit-learn tensorflow keras

Author : Nyanjui
Date   : 05 September 2026
"""


# ================================================================================
# SECTION 0: Import the required modules
# ================================================================================
import matplotlib.pyplot as plt
import numpy as np
import warnings
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import  Sequential
from tensorflow.keras.optimizers import Adam

# Suppress deprecation and future warnings for clean output
warnings.filterwarnings("ignore")

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
y = labels.astype(np.float32) # Cast to float32 for Tensorflow compatibility

# Split the data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ================================================================================
# SECTION 2: Model (neural network) initialisation, training, prediction, and evaluation
# ================================================================================
# Scale the data (NB: Neural Networks work better with scaled data)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network
model = Sequential([
   Dense(10, activation='relu', input_shape=(2,)),     # Input layer with 2 features
   Dense(5, activation='relu'),                        # Hidden layer
   Dense(1, activation='sigmoid')                      # Output layer for binary classification
])

# Compile the model
model.compile(
   optimizer=Adam(learning_rate=0.01),
   loss='binary_crossentropy',
   metrics=['accuracy']
)

# Train the model
history = model.fit(
   X_train,y_train,
   epochs=50,
   batch_size=10,
   validation_split=0.2,
   verbose=1
)

# Evaluate the model
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > .5).astype(int).reshape(-1)

# Evaluation metrics
accuracy = accuracy_score(y_test, y_pred_classes)
conf_matrix = confusion_matrix(y_test, y_pred_classes)
class_report = classification_report(y_test, y_pred_classes, target_names=['Apples', 'Oranges'])

# ================================================================================
# SECTION 3: Results and visualisation
# ================================================================================
# Display results
print(f"Accuracy of the Neural Network Model: {accuracy:.2f}")
print(f"\nConfusion Matrix of the Neural Network Model:\n{conf_matrix}")
print(f"\nClassification Reports of the Neural Network Model:\n{class_report}")

# Plot training and validation accuracy
plt.figure(figsize=(10,5))
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()