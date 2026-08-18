"""
================================================================================
Python script to demonstrate data integration and agreggation
================================================================================


Requirements
--------------------------
 pip install pandas

Author : Nyanjui
Date   : 18 August 2026
"""
# ================================================================================
# STEP 0: Import the required modules
# ================================================================================
import pandas as pd

# ================================================================================
# STEP 1: Create 2 datasets that are to be merged
# ================================================================================
# a)  Customer Information Dataset
customers = {
   'CustomerID' : [1,2,3],
   'Name': ['Aaron', 'Brenda', "Charlie"]
}
# Create the customer information dataframe
df_customers = pd.DataFrame(customers)

# b) Customer Transaction Dataset
transactions = {
   'CustomerID' : [1,1,2,3,3],
   'TransactionAmount' : [200, 150, 300, 500, 250],
   'TransactionDate' : ['2024-01-01','2024-01-15','2024-02-01','2024-03-01','2024-03-10', ]
}

# Create the transaction dataframe
df_transactions = pd.DataFrame(transactions)

# Display the original dataset
print(f"Customer Information Dataset")
print("-" * 55)
print(df_customers)
print("-" * 55)
print(f"Transaction Details Dataset")
print("-" * 55)
print(df_transactions)
print("-" * 55)

# ================================================================================
# STEP 2: Integrate/merge the customer and transaction details datasets on a
# common field ('CustomerID')
# ================================================================================
df = pd.merge(df_customers, df_transactions, on='CustomerID')
print(f"\nMerged customer and transaction details dataset:\n{df}")

# ================================================================================
# STEP 3: Aggregate the data -> Calculate the total amount spent by each customer
# ================================================================================
df_aggregated = df.groupby('CustomerID')['TransactionAmount'].sum().reset_index()

# Display the aggregated dataset
print(f"\nTotal transaction amount per customer :\n{df_aggregated}")

