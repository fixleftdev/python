import numpy as np
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([[1], [2], [3], [4], [5]])

y = np.array([1, 3, 2, 3, 5])


# Create a linear regression model with normalize parameter k

model = LinearRegression(normalize=True)

# Fit the model
model.fit(X, y)


# Make predictions
predictions = model.predict(X)

# Print the coefficients

print("Coefficients:", model.coef_)

print("Intercept:", model.intercept_)

print("Predictions:", predictions)

