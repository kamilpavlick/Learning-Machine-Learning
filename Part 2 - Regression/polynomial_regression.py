# Polynomial Regression

import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('polynomial_regression.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)

from sklearn.linear_model import LinearRegression
poly_reg.fit(X_poly, y)
model = LinearRegression()
model.fit(X_poly, y)


# Visualising the Polynomial Regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, model.predict(X_poly), color = 'blue')
plt.title('Polynomial Regression Example')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
