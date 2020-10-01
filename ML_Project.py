import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from joblib import dump,load
from joblib import dump,load
import numpy as np


housing=pd.read_csv('data.csv')
housing.head() #top 5 row
housing.info() #information,if missing data ,so take decision according to missing data
housing['CHAS'].value_counts() #info about data,how much value
housing.describe() #complete detail about data

# matplotlib inline
housing.hist(bins=50,figsize=(20,15))
plt.show()

#train test splitting
# For learn
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]

#train_set, test_set = split_train_test(housing, 0.2)
# print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")
train_set,test_set=train_test_split(housing,test_size=0.2,random_state=42)
print(f"Rows in train set: {len(train_set)}\nRows in test set: {len(test_set)}\n")
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

    # strat_test_set['CHAS'].value_counts()
    # strat_train_set['CHAS'].value_counts()
    # housing = strat_train_set.copy()


print(strat_test_set['CHAS'].value_counts())
print(strat_train_set['CHAS'].value_counts())

housing=strat_train_set.copy()

#corelation
corr_matrix=housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)

attributes=['RM',"ZN","MEDV","LSTAT"]
scatter_matrix(housing[attributes],figsize=(12,8))
housing.plot(kind="scatter", x="RM", y="MEDV", alpha=0.8)

#attribute combination
housing['TAXRM']=housing['TAX']/housing['RM']
housing.head()

corr_matrix=housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)

housing.plot(kind="scatter", x="TAXRM", y="MEDV", alpha=0.8)

housing=strat_train_set.drop("MEDV",axis=1)
housing_lables=strat_train_set["MEDV"].copy()


#missing attributes to take care missing attributes
#1-get rid of the missing
#2. get rid of the whole attribute
#3. set value (0 or mean or median)

a=housing.dropna(subset=['RM']) #opt 1
print(a.shape)

housing.drop('RM',axis=1) #option 2 drop column
housing.drop('RM',axis=1).shape
median=housing["RM"].median()

housing['RM'].fillna(median) #fiil median in blank cell #opt3
housing.describe()

imputer=SimpleImputer(strategy='median')
imputer.fit(housing)
print(imputer.statistics_)
print(imputer.statistics_.shape)
X=imputer.transform(housing)
housing_tr=pd.DataFrame(X,columns=housing.columns)
housing_tr.describe()

#creating pipeline
my_pipeline=Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('std_scalar',StandardScaler()),
                    ])
housing_num_tr=my_pipeline.fit_transform(housing)
print(housing_num_tr.shape)

## select model for the data

# model=LinearRegression()
# model=DecisionTreeRegressor()
model=RandomForestRegressor()
model.fit(housing_num_tr,housing_lables)
some_data=housing.iloc[:5]
some_labels=housing_lables.iloc[:5]
prepared_data=my_pipeline.transform(some_data)
model.predict(prepared_data)
list(some_labels)

#evaluating the model
housing_predictions=model.predict(housing_num_tr)
mse=mean_squared_error(housing_lables,housing_predictions)
rmse=np.sqrt(mse)
print(rmse)

#using better evalution technique -Cross validation
scores=cross_val_score(model,housing_num_tr,housing_lables,scoring="neg_mean_squared_error",cv=10)
rmse_scores=np.sqrt(-scores)
print(rmse_scores)

def print_scores(scores):
    print("Scores:",scores)
    print("mean :",scores.mean())
    print("Standard deviation:",scores.std())
    print_scores(rmse_scores)

#create model
dump(model,'real_estates_data.joblib')

#model testing
X_test=strat_test_set.drop("MEDV",axis=1)
Y_test=strat_test_set["MEDV"].copy()
X_test_prepared=my_pipeline.transform(X_test)
final_predictions=model.predict(X_test_prepared)
final_mse=mean_squared_error(Y_test,final_predictions)
final_rmse=np.sqrt(final_mse)
print(final_predictions)
print("**********")
print(list(Y_test))
print(final_rmse)
print(prepared_data[0])

model=load('real_estates_data.joblib')
features=np.array([-0.43942006,  33.12628155, -1.12165014, -0.27288841, -1.42262747,
       -0.24535958, -1.31238772,  22.61111401, -1.0016859 , -0.5778192 ,
       -0.97491834,  10.41164221, -0.86091034])
print(model.predict([features]))