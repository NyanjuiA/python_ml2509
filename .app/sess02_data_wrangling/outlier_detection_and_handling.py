"""
================================================================================
Python script to detect and handle outliers
================================================================================
This script demostrates how to detect and handle outliers using the
IQR(interquartile range)


Requirements
--------------------------
 pip install numpy pandas

Author : Nyanjui
Date   : 15 August 2026
"""
# ================================================================================
# SECTION 0: Import the required modules
# ================================================================================
import numpy as np
import pandas as pd

# ================================================================================
# STEP 1: Create a sample dataset
# ================================================================================
data = {
   'Name': ["Abigail", "Kamau", "Sharlene", "Diana", 'Mueni', "Frank","Grace"],
   'Age' : [5, 35, 32, 29, 45, 120, 28], # 5 and 120 are outliers
   'Salary': [50000, 2500, 54000, 52000, 110000,47000, 51000], # 2500 and 110000 are outliers
}

# ================================================================================
# STEP 2: Convert the above dictionary into a dataframe and display it
# ================================================================================
df = pd.DataFrame(data)
print(f"Original Dataframe:\n{df}")

# ================================================================================
# STEP 3: Detect 'age' and 'salary' outliers and display them
# ================================================================================
Q1 = df['Age'].quantile(.25) # First quartile (Q1)
Q3 = df['Age'].quantile(.75) # Third quartile (Q3)
IQR = Q3 - Q1 # Get the interquartile range for age
# Define the 'Age' outlier boundaries
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# TODO: Detect outliers using IQR method for the 'Salary' column


# Identify and display the outliers for the 'Age' column
outliers = df[(df['Age'] < lower_bound) | (df['Age'] > upper_bound)]
print(f"Detected outliers for the 'Age' column:\n{outliers}")

# ================================================================================
# STEP 4: Handle outliers -> Method i) Remove/drop outliers
# ================================================================================
df_no_age_outliers = df[(df['Age'] >= lower_bound) & (df['Age'] <= upper_bound)]
# Display the dataset after removing outliers for the age column
print(f"Dataset after removing outliers for the 'Age' column:\n{df_no_age_outliers}")

# ================================================================================
# STEP 5: Handle outliers -> Method ii) Cap outliers with boundary values
# ================================================================================
df['Age'] = np.where(df['Age'] < lower_bound,lower_bound,
                     np.where(df['Age'] > upper_bound,upper_bound,df['Age']))
# Display the dataset after capping outliers for the 'Age' column
print(f"Dataset after capping the outliers for the 'Age' column:\n{df}")