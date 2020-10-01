import numpy as np
import  matplotlib.pyplot as plt
import sklearn
from sklearn import  datasets,linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error

iris=datasets.load_iris()
features=iris.data
print(iris.DESCR)
labels = iris.target
print(features[0],labels[0])

#traing the classifier
clasifier_va=KNeighborsClassifier()
clasifier_va.fit(features,labels)
preds=clasifier_va.predict([[5.1,3.5,1.4,0.2]])
print(preds)