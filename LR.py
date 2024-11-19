# Import the necessary libraries
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Import the dataset
dataset = pd.read_csv("/Users/adityadebchowdhury/Desktop/Desktop - Aditya’s MacBook Air/PROGRAMMING 2/IDA ML/salary_data.csv")

# Split into independent (X) and dependent (y) variables
X = dataset.iloc[:, [0]].values  # Get Years of Experience column
y = dataset.iloc[:, [1]].values  # Get Salary column

# Split the dataset into training and testing sets (1/3 of the data for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# Create a linear regression model
regressor = LinearRegression()

# Train the model using the training data
regressor.fit(X_train, y_train)

# Predict the salaries for the test data
y_pred = regressor.predict(X_test)

# Plot the training data and the linear regression line
plt.scatter(X_train, y_train, color='red')  # Plot the actual training points
plt.plot(X_train, regressor.predict(X_train), color='blue')  # Plot the regression line
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Plot the test data and the regression line
plt.scatter(X_test, y_test, color='green')  # Plot the actual test points
plt.plot(X_train, regressor.predict(X_train), color='blue')  # Regression line (same as above)
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Print R-squared score to evaluate the model
r2 = r2_score(y_test, y_pred)
print("R-squared score:", r2)