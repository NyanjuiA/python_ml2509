"""
================================================================================
Python script to demonstrate Pandas (Panel Data) Dataframes.
================================================================================
This script demonstrates how to create, display and explore Pandas Dataframes

Requirements
--------------------------
 pip install pandas numpy

Author : Nyanjui
Date   : 14/08/2026
"""
# ================================================================================
# SECTION 0: Import the required modules
# ================================================================================
import numpy as np
import pandas as pd

# ================================================================================
# SECTION 1: Create a dataframe from a dictionary
# ================================================================================
print("---- 1. Create a dataframe from a dictionary ----")
data = {
   'Name': ["Abigail", "Kamau", "Sharlene", "Diana", 'Mueni'],
   'Age' : [25, 30, 35, 28, 32],
   'City': ["Nakuru", "Limuru", "Kisumu", "Homabay", 'Makueni'],
   'Salary': [55000, 65000, 72000, 48000, 60000],
   'Department': ['HR', "IT", 'IT', "Marketing", "Finance"]
}

# ================================================================================
# SECTION 2: Create and display a pandas dataframe from above dictionary
# ================================================================================
print("---- 2. Create and display employees dataframe from above dictionary ----")
df = pd.DataFrame(data)
print(f"{df}")
print("-" * 50)

# ================================================================================
# SECTION 3: Create and display another dataframe with a custom index
# ================================================================================
print("---- 3. Create a dataframe with a custom index from a dictionary ----")
df_indexed = pd.DataFrame(data, index=['Emp1', 'Emp2', 'Emp3', 'Emp4', 'Emp5'])
print(f"Employee details indexed dataframe:\n{df_indexed}")
print("-" * 50)

# ================================================================================
# SECTION 4: Dataframe attributes and information
# ================================================================================
print("---- 4. Display dataframe attributes and information ----")
print(f"First 3 rows of the employee details using 'head()' function: \n{df.head(3)}")
print(f"Last 2 rows of the employee details using 'tail()' function: \n{df.tail(2)}")
print("-" * 50)

# ================================================================================
# SECTION 5: Access data from a dataframe
# ================================================================================
print("---- 5. Access data from a dataframe ----")
print(f"Single column (Series):\n{df['Name']}")
print(f"Multiple columns :\n{df[['Name','Age','Department']]}")
print(f"\nAccess by index location/position:\nFirst row:\n{df.iloc[0]}"
      f"\nSpecific cell (row 2, column 'Age'):\n{df.iloc[1,1]}")
print(f"Access by label (loc):\nEmployee with index 2:\n{df.loc[2]}")
print("-" * 50)

# ================================================================================
# SECTION 6: Create a dataframe from a list of lists
# ================================================================================
print("---- 6. Create a datafrme from a list of lists ----")
product_data = [
    ['Laptop', 99999.5, 'Electronics', 50],
    ['Mouse', 250.0, 'Electronics',200],
    ['Notebook',599.0, 'Stationary', 150],
    ['Pen',199.0,'Stationary',500]
]
product_columns = ['Product','Price','Category','Stock']
product_df = pd.DataFrame(product_data, columns=product_columns)
print(f"Product DataFrame:\n{product_df}")
print("-" * 50)