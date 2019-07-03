# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('multiple_linear_regression.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

X = pd.get_dummies(pd.DataFrame(X), columns=[3]).values

# Avoiding the Dummy Variable Trap
X = X[:, :-1]

pd.plotting.scatter_matrix(dataset)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Checking p-value for our variables

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(float), values = X, axis = 1)
X_opt = (X[:, [0, 1, 2, 3, 4, 5]]).astype(float)
regressor_OLS = sm.OLS(endog=y, exog=X_opt, missing='drop').fit()
regressor_OLS.summary()

X_opt = X[:, [0, 1, 3, 4, 5]].astype(float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 4, 5]].astype(float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3, 5]].astype(float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

X_opt = X[:, [0, 3]].astype(float)
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()