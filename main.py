import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()
#print(diabetes.keys())
# dict_keys(['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename'])
#print(diabetes.DESCR)

#diabetes_X=diabetes.data[:,np.newaxis,2]
diabetes_X = diabetes.data

# print(diabetes_X)
diabetes_X_train = diabetes_X[:-30]  # last 30 features
diabetes_X_test = diabetes_X[-30:]  # first 20

diabetes_Y_train = diabetes.target[:-30]  # labels
diabetes_Y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()
model.fit(diabetes_X_train, diabetes_Y_train)
diabetes_Y_predicted = model.predict(diabetes_X_test)
print("mean", mean_squared_error(diabetes_Y_test, diabetes_Y_predicted))
print("weights", model.coef_)
print("intercept", model.intercept_)
plt.scatter(diabetes_X_test,diabetes_Y_test)
plt.plot(diabetes_X_test,diabetes_Y_predicted)
plt.show()


