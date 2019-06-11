#   1.importing libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


#   2.importing CSV and prepare x and y if the y is last variable of vector

dataset = pd.read_csv("data.csv")

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values


#   3. taking care of missing data

from sklearn.impute import SimpleImputer

simpleimputer = SimpleImputer()  # use default methods.
X[:, 1:3] = simpleimputer.fit_transform(X[:, 1:3])


#   4. Encode categorical data
#   https://stackoverflow.com/questions/54160370/how-to-use-sklearn-column-transformer

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import make_column_transformer


labelencoder_X = LabelEncoder()
#X[:, 0] = labelencoder_X.fit_transform(X[:, 0])


preprocessor = make_column_transformer( (OneHotEncoder(), [0]), remainder="passthrough")
X = preprocessor.fit_transform(X)

labelencoder_Y = LabelEncoder()
y[:, 0] = labelencoder_Y.fit_transform(y[:, 0])

#   5. Splitting the test for the training and the test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


#   6. Feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
# For test set we only use transform, because scaler was fitted to the training set
X_test = sc_X.transform(X_test)




