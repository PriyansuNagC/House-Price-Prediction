#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


housing = pd.read_csv("data.csv")


# In[3]:


housing.head()


# In[4]:


housing.info()


# In[5]:


housing.describe()


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


import matplotlib.pyplot as plt


# In[8]:


housing.hist(bins=50 , figsize=(20 , 15))


# In[9]:


import numpy as np
def train_test(data , testR) :
    np.random.seed(42)
    shuffle = np.random.permutation(len(data))
    testSS = int(len(data)*testR)
    testI = shuffle[:testSS]
    trainI = shuffle[testSS :]
    return data.iloc[trainI] , data.iloc[testI]


# In[10]:


#trainSet , testSet = train_test(housing , 0.2)


# In[11]:


#print(f"Rows in train set : {len(trainSet)}\nRows in test set : {len(testSet)}")


# In[12]:


from sklearn.model_selection import train_test_split
trainSet , testSet = train_test_split(housing , test_size=0.2 , random_state=42)
print(f"Rows in train set : {len(trainSet)}\nRows in test set : {len(testSet)}")


# In[13]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1 , test_size=0.2 , random_state=42)
for trainI , testI in split.split(housing , housing['distance 2']) :
    stratTrain = housing.loc[trainI]
    stratTest = housing.loc[testI]


# In[14]:


stratTest.info()


# In[15]:


corrM = housing.corr()
corrM['age'].sort_values(ascending=False)


# In[16]:


from pandas.plotting import scatter_matrix


# In[17]:


attr = ["age","date","price"]
scatter_matrix(housing[attr], figsize=(12,8))


# In[18]:


housing.plot(kind="scatter",x="price",y="age",alpha=0.8)


# In[19]:


housing = stratTrain.drop("age",axis=1)
Label = stratTrain["age"].copy()


# In[20]:


from sklearn.impute import SimpleImputer  # to fill missing values in the dataset
imputer = SimpleImputer(strategy = "median")
imputer.fit(housing)


# In[21]:


imputer.statistics_.shape


# In[22]:


x = imputer.transform(housing)
hTr = pd.DataFrame(x , columns=housing.columns)
hTr.describe()


# In[23]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
myPipe = Pipeline([
    ('imputer',SimpleImputer(strategy="median")),
    ('std_scalar',StandardScaler())
])


# In[24]:


housingNum = myPipe.fit_transform(hTr)


# In[25]:


housingNum.shape


# In[26]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(housingNum,Label)


# In[27]:


sData = housing.iloc[:5]
sLabel = Label.iloc[:5]
prData = myPipe.transform(sData)
model.predict(prData)


# In[28]:


list(sLabel) 


# In[29]:


from sklearn.metrics import mean_squared_error
housingPr = model.predict(housingNum)
mse = mean_squared_error(Label , housingPr)
r = np.sqrt(mse)
r


# In[30]:


from sklearn.model_selection import cross_val_score
score = cross_val_score(model , housingNum , Label , scoring ="neg_mean_squared_error", cv=10)
r_score = np.sqrt(-score)
r_score


# In[31]:


def pr_score(score) :
    print("SCORES : ",score)
    print("MEAN : ", score.mean())
    print("STANDARD DEVIATION : ", score.std())
pr_score(r_score)


# In[32]:


from joblib import dump , load
dump(model,'dragon.joblib')


# In[33]:


x_test = stratTest.drop("age",axis=1)
y_test = stratTest["age"].copy()
xtest_pr = myPipe.transform(x_test)
final_pr = model.predict(xtest_pr)
final_mse = mean_squared_error(y_test , final_pr)
final_rmse = np.sqrt(final_mse)
final_rmse


# In[34]:


print(final_pr , list(y_test))


# In[35]:


prData[0]


# In[ ]:




