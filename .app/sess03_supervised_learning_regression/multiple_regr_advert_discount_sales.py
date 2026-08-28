"""
================================================================================
Python script to predict the sales for the year 2024 using multiple linear regression
using advertising amount($) and discount(%) to determine sales
================================================================================


Requirements
--------------------------
 pip install matplotlib pandas scikit-learn

Author : Nyanjui
Date   : 27 August 2026
"""
# ================================================================================
# SECTION 0: Import the required modules
# ================================================================================
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# ================================================================================
# SECTION 1: Create a dictionary and convert it to pd dataframe from data in sess03 slides
# ================================================================================
data = {
   'Year': [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,
            2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019],
   'Advertising Budget ($)': [150, 200, 180, 220, 170, 250, 210, 160, 240, 230,
                              190, 220, 260, 250, 200, 230, 180, 240, 190, 220],
   'Discounts Given (%)': [5, 10, 7, 8, 6, 12, 9, 5, 10, 9, 7, 11, 15, 14, 8, 10,
                           6, 12, 8, 9],
   'Sales ($)': [1050, 1400, 1320, 1550, 1200, 1750, 1600, 1150, 1700, 1650,
                 1350, 1600, 1850, 1800, 1450, 1650, 1300, 1700, 1400, 1550]
}

# Convert the above dictionary into a pd dataframe
df = pd.DataFrame(data)

# Split the independent variable (X) and dependent variable (y)
X = df[['Advertising Budget ($)', 'Discounts Given (%)']] # Independent variables
y = df['Sales ($)'] # Dependent variable

# ================================================================================
# SECTION 3: Create, train and make predictions using linear regression model
# ================================================================================
model = LinearRegression()

model.fit(X, y) # Train the model

# Predict sales for 2024 with an advertising budget of $250 and discount of 12%
budget_2024 = 250
discount_2024 = 12
sales_2024 = model.predict([[budget_2024, discount_2024]])

# Display the predictions
print(f"Predicted sales for 2024 with an advertising budget of $ {budget_2024} "
      f"and a discount of {discount_2024}5 are $ {sales_2024}.")

# ================================================================================
# SECTION 4: Visualise the data
# ================================================================================
plt.figure(figsize=(10, 6))

# Plot the actual sales
plt.scatter(df['Year'], df['Sales ($)'], color='blue', label='Actual Sales', marker='o')

# Plot predictions for the data points
y_pred = model.predict(X)
plt.plot(df['Year'], y_pred, color='green', label='Predicted Sales Line')
plt.scatter(2024,sales_2024, color='red', label='Predicted Sales for 2024', marker='x', s=100)

# Add the labels and titles
plt.xlabel('Year')
plt.ylabel('Sales ($)')
plt.title(f"Predicted sales for 2024 based on an advertising budget and discounts ")
plt.legend()
plt.show() # Display/show the plot

