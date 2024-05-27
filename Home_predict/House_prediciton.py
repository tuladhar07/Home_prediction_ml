#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
matplotlib.rcParams["figure.figsize"]=(20,10)


# In[2]:


df=pd.read_csv('bengaluru_house_prices.csv')
df.head()


# In[3]:


df.shape


# In[4]:


df.groupby('area_type')['area_type'].agg('count')


# In[5]:


df2=df.drop(['area_type', 'society', 'balcony', 'availability'], axis='columns')
df2.head()


# In[6]:


df2.isnull().sum()


# In[7]:


df3=df2.dropna()
df3.isnull().sum()


# ## since the BHK is number + text we will create a new column where we will just take its numerical value

# In[9]:


df3['BHK']=df3['size'].apply(lambda x: int(x.split(' ')[0]))


# In[10]:


df3.isnull().sum()


# In[12]:


df3['BHK'].unique()


# In[13]:


df3[df3.BHK>20]


# In[14]:


df3.total_sqft.unique()


# In[15]:


def convert_sqft_to_num(x):
    token=x.split('-')
    if len(token)==2:
        return (float(token[0])+float(token[1]))/2
    try:
        return float(x)
    except:
        return None


# In[17]:


convert_sqft_to_num('2551')


# In[19]:


df4=df3.copy()
df4['total_sqft']=df4['total_sqft'].apply(convert_sqft_to_num)
df4.head(20)


# In[20]:


df5=df4.copy()


# ## Feature Engineering

# In[21]:


df5['price_per_sqft']=df5['price']*100000/df5['total_sqft']
df5.head(10)


# In[22]:


len(df5.location.unique())


# In[26]:


df5.location=df5.location.apply(lambda x: x.strip())
location_stats=df5.groupby('location')['location'].agg('count').sort_values(ascending=False)
location_stats


# In[35]:


location_stats_less_than_10=(location_stats[location_stats<=10])


# In[36]:


len(df5.location.unique())


# In[37]:


df5.location=df5.location.apply(lambda x:'other' if x in location_stats_less_than_10 else x)
len(df5.location.unique())


# In[39]:


df5.head(20)


# In[41]:


df6=df5[~(df5.total_sqft/df5.BHK<300)]
df6.shape


# In[43]:


def remove_pps_outliers(df):
    df_out= pd.DataFrame()
    for key, subdf in df.groupby('location'):
        m=np.mean(subdf.price_per_sqft)
        st=np.std(subdf.price_per_sqft)
        reduced_df=subdf[(subdf.price_per_sqft>(m-st))& (subdf.price_per_sqft<=(m+st))]
        df_out=pd.concat([df_out, reduced_df], ignore_index=True)
    return df_out

df7=remove_pps_outliers(df6)
df7.shape


# In[45]:


def plot_scatter_chart(df,location):
    BHK2 = df[(df.location==location) & (df.BHK==2)]
    BHK3 = df[(df.location==location) & (df.BHK==3)]
    matplotlib.rcParams['figure.figsize'] = (15,10)
    plt.scatter(BHK2.total_sqft,BHK2.price,color='blue',label='2 BHK', s=50)
    plt.scatter(BHK3.total_sqft,BHK3.price,marker='+', color='green',label='3 BHK', s=50)
    plt.xlabel("Total Square Feet Area")
    plt.ylabel("Price (Lakh Indian Rupees)")
    plt.title(location)
    plt.legend()
    
plot_scatter_chart(df7,"Rajaji Nagar")


# In[46]:


plot_scatter_chart(df7,"Hebbal")


# ## We should also remove properties where for same location, the price of (for example) 3 bedroom apartment is less than 2 bedroom apartment (with same square ft area). What we will do is for a given location, we will build a dictionary of stats per BHK, i.e.
# 
# {
#     '1' : {
#         'mean': 4000,
#         'std: 2000,
#         'count': 34
#     },
#     '2' : {
#         'mean': 4300,
#         'std: 2300,
#         'count': 22
#     },    
# }
# Now we can remove those 2 BHK apartments whose price_per_sqft is less than mean price_per_sqft of 1 BHK apartment

# In[48]:


def remove_BHK_outliers(df):
    exclude_indices = np.array([])
    for location, location_df in df.groupby('location'):
        BHK_stats = {}
        for BHK, BHK_df in location_df.groupby('BHK'):
            BHK_stats[BHK] = {
                'mean': np.mean(BHK_df.price_per_sqft),
                'std': np.std(BHK_df.price_per_sqft),
                'count': BHK_df.shape[0]
            }
        for BHK, BHK_df in location_df.groupby('BHK'):
            stats = BHK_stats.get(BHK-1)
            if stats and stats['count']>5:
                exclude_indices = np.append(exclude_indices, BHK_df[BHK_df.price_per_sqft<(stats['mean'])].index.values)
    return df.drop(exclude_indices,axis='index')
df8 = remove_BHK_outliers(df7)
# df8 = df7.copy()
df8.shape


# In[49]:


plot_scatter_chart(df8,"Rajaji Nagar")


# In[50]:


plot_scatter_chart(df8,"Hebbal")


# In[51]:


df8.bath.unique()


# In[53]:


df8[df8.bath>10]
df8[df8.bath>df8.BHK+2]


# In[54]:


df9 = df8[df8.bath<df8.BHK+2]
df9.shape


# In[55]:


df10 = df9.drop(['size','price_per_sqft'],axis='columns')
df10.head(3)


# In[57]:


dummies = pd.get_dummies(df10.location)
dummies.head()


# In[62]:


df11=pd.concat([df10, dummies.drop('other', axis='columns')], axis='columns')
df11.head(3)


# In[64]:


df12=df11.drop('location', axis='columns')


# In[65]:


x=df12.drop('price', axis='columns')
x.head()


# In[66]:


y=df12.price


# In[69]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)


# In[71]:


from sklearn.linear_model import LinearRegression
lr_clf=LinearRegression()
lr_clf.fit(x_train, y_train)
lr_clf.score(x_test, y_test)


# In[73]:


from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score

cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

cross_val_score(LinearRegression(), x, y, cv=cv)


# In[79]:


from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor

def find_best_model_using_gridsearchcv(x,y):
    algos = {
        'linear_regression' : {
            'model': LinearRegression(),
            'params': {
                'copy_X' : [True, False],
                'fit_intercept' : [True, False],
                'n_jobs' : [1,2,3],
                'positive' : [True, False]
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
        gs.fit(x,y)
        scores.append({
            'model': algo_name,
            'best_score': gs.best_score_,
            'best_params': gs.best_params_
        })

    return pd.DataFrame(scores,columns=['model','best_score','best_params'])

find_best_model_using_gridsearchcv(x,y)


# In[81]:


def predict_price(location,sqft,bath,bhk):    
    loc_index = np.where(X.columns==location)[0][0]

    x = np.zeros(len(X.columns))
    x[0] = sqft
    x[1] = bath
    x[2] = bhk
    if loc_index >= 0:
        x[loc_index] = 1

    return lr_clf.predict([x])[0]


# ## Export the tested model to a pickle file

# In[82]:


import pickle
with open('banglore_home_prices_model.pickle','wb') as f:
    pickle.dump(lr_clf,f)


# ## Export location and column information to a file that will be useful later on in our prediction application

# In[85]:


import json
columns = {
    'data_columns' : [col.lower() for col in x.columns]
}
with open("columns.json","w") as f:
    f.write(json.dumps(columns))


# In[ ]:




