#!/usr/bin/env python
# coding: utf-8

# # This model predicts the price of Bangalore's house with the help of a few parameters like availability, size, total square feet, bath, location, etc. 

# In[1]:


from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import plotly.express as px


# In[2]:


df = pd.read_csv("bengaluru_house_prices.csv")


# In[3]:


df


# In[4]:


df.shape


# In[5]:


df.nunique()


# In[6]:


df.area_type.value_counts()


# In[7]:


df.isnull().sum()


# In[8]:


#Society column has higher number of data with null values so dropping it from model


# In[9]:


df = df.drop(["society"], axis = 1)


# In[10]:


df.balcony.unique()


# In[11]:


df.balcony.value_counts()


# ### Most Repetative number in a balcony is 2, so replacing null value with 2.0

# In[12]:


df['balcony'] = df['balcony'].replace("nan", 2.0)


# In[13]:


df['balcony'].unique()


# In[14]:


df.isnull().sum()


# # All other columns has very few null values so we dropping all null values

# In[15]:


df = df.dropna()


# In[16]:


df.isnull().sum()


# In[17]:


#Now the data is without any null valus


# In[18]:


df


# In[19]:


df.availability.value_counts()


# In[20]:


df.info()


# In[21]:


df['size']  = df['size'].astype(str)


# In[22]:


df.info()


# In[23]:


df['size'].unique()


# In[24]:


df['size'] = df['size'].apply(lambda x: int(x.split(" ")[0]))


# In[25]:


df['size'].unique()


# In[26]:


df['bhk'] = df['size']


# In[27]:


df


# In[28]:


df.total_sqft.unique()


# ### we need to fix this 1133 - 1384 this kind of range numbers as above

# In[29]:


def is_float(x):
    try:
        float(x)
    except:
        return False
    return True


# In[30]:


df[~df["total_sqft"].apply(is_float)].head(20)


# In[31]:


def convert_sqrt_to_num(x):
    tokens = x.split("-")
    if len(tokens) == 2:
        return(float(tokens[0])+ float(tokens[1]))/2
    try:
        return float(x)
    except:
        return None


# In[32]:


convert_sqrt_to_num("1128-1864")


# In[33]:


df1 = df.copy()


# In[34]:


df1.total_sqft = df1.total_sqft.apply( convert_sqrt_to_num)


# In[35]:


df1


# In[36]:


df2 = df1.copy()


# In[37]:


#Creating a new column for price_per_sq


# In[38]:


df2["price_per_sqft"] = df2.price*100000/df2.total_sqft


# In[39]:


df2


# In[40]:


df2.location.unique()


# In[41]:


df2.location.value_counts()


# In[42]:


loc = df2.groupby("location")["location"].agg("count").sort_values(ascending = False)


# In[43]:


loc


# In[44]:


len(loc[loc<10])


# In[45]:


less_then_10 = loc[loc<=10]


# In[46]:


less_then_10


# In[47]:


df2["location"] = df2.location.apply(lambda x: "other place" if x in less_then_10 else x)


# In[48]:


df2.location.unique()


# In[49]:


df2


# # Checking for outlayer is total_sqft and bath

# In[50]:


df2[df2["total_sqft"]/df2["bhk"]<300]


# In[51]:


df2.shape


# ### Removing the outliers of price_per_sqft from dataframe

# In[52]:


df3 = df2[~(df2["total_sqft"]/df2["bhk"]<300)]


# In[53]:


df3.groupby("location").mean()


# In[54]:


def remove_pps_outliers(df3):
    df_out = pd.DataFrame()
    for key, subdf in df3.groupby('location'):
        m = np.mean(subdf.price_per_sqft)
        st = np.std(subdf.price_per_sqft)
        reduced_df = subdf[(subdf.price_per_sqft>(m-st)) & (subdf.price_per_sqft<=(m+st))]
        df_out = pd.concat([df_out,reduced_df],ignore_index=True)
    return df_out
df4 = remove_pps_outliers(df3)
df4.shape


# In[ ]:





# In[55]:


df4.shape


# In[56]:


df4.describe()


# In[57]:


import seaborn as sns
sns.boxplot(df4.price_per_sqft,saturation=4,
    width=1)


# In[ ]:





# In[58]:


df4.price_per_sqft.describe()


# In[ ]:





# In[59]:


df11 = df4


# In[60]:


df11


# In[61]:


df4.corr()


# In[62]:


df4.head()


# In[63]:


df.area_type.value_counts()


# In[64]:


from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import chi2, chi2_contingency
import scipy
base = ols("price  ~ area_type ", data = df4).fit()
sm.stats.anova_lm(base)


# In[65]:


Heating_Typ = pairwise_tukeyhsd(df4.price,df4.area_type,alpha=0.05)
print(Heating_Typ)


# # As P value is above 0.05, We can use this in model

# In[66]:


df4


# In[67]:


df.availability.value_counts()


# In[68]:


# As ready to move dominates the entir colum their will be no import of adding availability in the model


# In[69]:


df4


# In[70]:


# Dropping  availability and price_per_sqft from the model


# In[71]:


df5 = df4.drop(["availability","price_per_sqft","size","area_type"], axis = 1)


# In[72]:


df5.corr()


# In[73]:


df5


# In[74]:


# Now our dataset is clean, so we will start encoding


# In[75]:


dummy = pd.get_dummies(df5.location)


# In[76]:


dummy.head(2)


# In[77]:


df6 = pd.concat([df5,dummy.drop("other place", axis="columns")], axis = "columns")
df6


# In[78]:


df6.shape


# In[79]:


df6.groupby("location")["bhk"].agg("count")


# In[80]:


x = df6.drop(["price","location"], axis = 1)


# In[81]:


x


# In[82]:


y = df6.price
y


# In[83]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,train_size=0.7)


# In[96]:


from sklearn.linear_model import LinearRegression
lm = LinearRegression(fit_intercept=True,n_jobs=200)
lm.fit(X_train,y_train)
print("Coefficient", lm.coef_ , "intercept", lm.intercept_)
print("The R_square is: ", round(lm.score (X_train,y_train),3))


# In[85]:


from sklearn.ensemble import RandomForestRegressor

RandomForest_model=RandomForestRegressor()
RandomForest_model.fit(X_train,y_train)
accuracy=RandomForest_model.score(X_test,y_test)
accuracy


# In[86]:


from sklearn.tree import DecisionTreeRegressor

DecisionTree_model=DecisionTreeRegressor()
DecisionTree_model.fit(X_train,y_train)
accuracy=DecisionTree_model.score(X_test,y_test)
accuracy


# In[87]:


from sklearn.svm import SVR
regressor = SVR(kernel = 'poly')
regressor.fit(X_train,y_train)
regressor.score(X_train, y_train)
regressor.score(X_test, y_test)


# In[88]:


X = x


# In[92]:


from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import ShuffleSplit
from sklearn import datasets, linear_model
from sklearn.model_selection import cross_val_score
def find_best_model_using_gridsearchcv(X,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'normalize': [True, False],"fit_intercept":[True, False]
            }
        },
        'lasso': {
            'model': Lasso(),
            'params': {
                'alpha': [1,2],
                'selection': ['random', 'cyclic']
            }
        },
        'decision_tree': {
            'model': DecisionTreeRegressor(),
            'params': {
                'criterion' : ['mse','friedman_mse'],
                'splitter': ['best','random']
            }
        }
    }
    scores = []
    cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
    for algo_name, config in algos.items():
        gs =  GridSearchCV(config['model'], config['params'], cv=cv, return_train_score=False)
        gs.fit(X,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(X,y)


# # linear_regression gives us highest accuracy of 82.2%

# In[ ]:




