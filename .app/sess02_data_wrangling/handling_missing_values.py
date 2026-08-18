"""
================================================================================
Python script to demonstrate handling missing values in data
================================================================================


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
   'Name': ["Abigail", "Kamau", "Sharlene", "Diana", 'Mueni'],
   'Age' : [24, np.nan, 22, 32, np.nan],
   'City': ["Nakuru", "Limuru", np.nan, "Homabay", 'Makueni'],
}

# ================================================================================
# STEP 2: Convert the above dictionary into a dataframe
# ================================================================================
df = pd.DataFrame(data)

# ================================================================================
# STEP 3: Display the original dataframe with missing values
# ================================================================================
print(f"Original DataFrame with missing values:\n{df}")

# ================================================================================
# STEP 4: Handle missing data/values
# ================================================================================
# OPTION 1: Drop rows with missing values
df_dropna = df.dropna()
# Display the dataset after dropping rows with missing values
print(f"Dataset after dropping missing values:\n{df_dropna}")

# OPTION 2: Impute/Fill in the missing values
# Fill in the mean for the missing ages in 'Age' column
df.fillna({'Age':df['Age'].mean()}, inplace=True)

# Fill in the missing categorical values(forward fill for the 'City' column)
df['City'] = df['City'].ffill()

# Display the dataset after imputing/filling in the missing values
print(f"Dataset after imputing/filling in missing values:\n{df}\n")