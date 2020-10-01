from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import numpy as np
import  matplotlib.pyplot as plt
import sklearn
from sklearn import  datasets,linear_model


iris=datasets.load_iris()
print(list(iris.keys()))
print(iris['data'])
print('****',iris['target'])

# to check whether flower is virginica or not
X=iris['data'][:,3:]
print(X)
y=(iris['target']==2).astype(np.int)
print(y)

# logistic regression classifier
log_clf=LogisticRegression()
log_clf.fit(X,y)
clf_pred=log_clf.predict(([[1.9]]))
print(clf_pred)

# matplot lib to visualisation
X_new=np.linspace(0,3,100).reshape(-1,1)
print(X_new)
y_probability=log_clf.predict_proba(X_new)
print(y_probability)
plt.plot(X_new,y_probability[:,1],"g-",Label="virginica")
plt.show()




