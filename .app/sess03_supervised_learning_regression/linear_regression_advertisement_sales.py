"""
================================================================================
Python script to predict sales for a given year(2019) based on advertisement and
sales data using simple linear regression
================================================================================


Requirements
--------------------------
 pip install matplotlib numpy scipy scikit-learn

Author : Nyanjui
Date   : 27 August 2026
"""

# ================================================================================
# SECTION 0: Import the required modules
# ================================================================================
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

# ================================================================================
# SECTION 1: Sample dara from sess03 slide 11 which will be reshaped into a 2-d array
# ================================================================================
years = np.array([2014,2015,2016,2017,2018])
advertisement = np.array([90,120,150,100,130])
sales = np.array([1000, 1300, 1800, 1200, 1380])

# Reshape the above data
year = years.reshape((-1,1))
advertisement = advertisement.reshape((-1,1))

# ================================================================================
# SECTION 2: Create a linear regression model and predict sales for 2019
# ================================================================================
model = LinearRegression()
model.fit(advertisement, sales) # Train the linear regression model

sales_2019 = model.predict([[200]]) # Use the model to predict sales for 2019 with
# an advertising budget of $200

# Display the predicted sales for 2019
print(f"Sales prediction for 2019 with an advertising budget of $200 : ${sales_2019}")

# ================================================================================
# SECTION 3: Visualise the sales and advertising data
# ================================================================================
# For plotting purposes, revert the advertising data back to 1-d array
advertisement = advertisement.reshape((1,-1)) # advertisement = np.array([90,120,150,100,130])

# Variables to plot or visualise the sales & advertising data
slope, intercept, r_value, p_value, std_err = stats.linregress(advertisement, sales)

# Define a function to calculate the regression model
def simple_regression(advertisement):
   return slope * advertisement + intercept

# Define the simple linear regression model
simple_regression_model = list(map(simple_regression, advertisement))

# Visualise the data
plt.scatter(advertisement, sales)
plt.plot(advertisement, simple_regression_model)
plt.title("Advertising vs. Sales")
plt.xlabel("Advertising Budget")
plt.ylabel("Sales")
plt.show()


