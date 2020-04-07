#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
from surprise import Reader
from surprise import Dataset
from surprise.model_selection import cross_validate
from surprise import NormalPredictor
from surprise import KNNBasic
from surprise import KNNWithMeans
from surprise import KNNWithZScore
from surprise import KNNBaseline
from surprise import SVD
from surprise import BaselineOnly
from surprise import SVDpp
from surprise import NMF
from surprise import SlopeOne
from surprise import CoClustering
from surprise.model_selection import KFold
from surprise.accuracy import rmse
from surprise import accuracy
from surprise.model_selection import train_test_split
from plotly.offline import init_notebook_mode, plot, iplot
import plotly.graph_objs as go
init_notebook_mode(connected=True)


# In[12]:


train_data = pd.read_csv("train.csv")
jokes_data = pd.read_csv("jokes.csv")
test_data = pd.read_csv("test.csv")
train_data = train_data[['user_id','joke_id','Rating']]


# In[13]:


df = pd.read_csv('train.csv')
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(df[['user_id','joke_id','Rating']], reader)
trainingSet = data.build_full_trainset()


# In[14]:


"""
Distribution of Ratings
"""
data = df['Rating'].value_counts().sort_index(ascending=False)
trace = go.Bar(x = data.index,
               text = ['{:.1f} %'.format(val) for val in (data.values / df.shape[0] * 100)],
               textposition = 'auto',
               textfont = dict(color = '#000000'),
               y = data.values,
               )
# Create layout
layout = dict(title = 'Distribution Of {} joke-ratings'.format(df.shape[0]),
              xaxis = dict(title = 'Rating'),
              yaxis = dict(title = 'Count'))
# Create plot
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# In[18]:


"""
Rating Distribution by Jokes
"""
# Number of ratings per joke
data = df.groupby('joke_id')['Rating'].count().clip(upper=150)
# configure_plotly_browser_state()
# init_notebook_mode(connected=False)
# Create trace
trace = go.Histogram(x = data.values,
                     name = 'Ratings',
                     xbins = dict(start = 0,
                                  end = 150,
                                  size = 2))
# Create layout
layout = go.Layout(title = 'Distribution Of Number of Ratings Per Joke (Clipped at 150)',
                   xaxis = dict(title = 'Number of Ratings Per Joke'),
                   yaxis = dict(title = 'Count'),
                   bargap = 0.2)

# Create plot
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# In[16]:


df.groupby('joke_id')['Rating'].count().reset_index().sort_values('Rating', ascending=False)[:10]


# In[17]:


"""
Rating Distribution by User
"""
# Number of ratings per user
data = df.groupby('user_id')['Rating'].count().clip(upper=150)
# configure_plotly_browser_state()
# init_notebook_mode(connected=False)

# Create trace
trace = go.Histogram(x = data.values,
                     name = 'Ratings',
                     xbins = dict(start = 0,
                                  end = 150,
                                  size = 2))
# Create layout
layout = go.Layout(title = 'Distribution Of Number of Ratings Per User (Clipped at 50)',
                   xaxis = dict(title = 'Ratings Per User'),
                   yaxis = dict(title = 'Count'),
                   bargap = 0.2)

# Create plot
fig = go.Figure(data=[trace], layout=layout)
iplot(fig)


# In[19]:


df.groupby('user_id')['Rating'].count().reset_index().sort_values('Rating', ascending=False)[:10]


# In[20]:


"""
Surprise Library
"""

df = pd.read_csv('train.csv')
reader = Reader(rating_scale=(0, 5))
data = Dataset.load_from_df(df[['user_id', 'joke_id', 'Rating']], reader)
trainingSet = data.build_full_trainset()


# In[21]:


df.sort_values(['Rating'],ascending=False).head()


# In[22]:


#test dataset is loaded using pandas read_csv
dt = pd.read_csv('test.csv')


# In[23]:


df.describe()


# In[24]:


"""
Building SVD Model and 
"""

algo=SVD(n_epochs=50,lr_all=0.01,reg_all =0.04,n_factors =250)

kf = KFold(n_splits=5)

for trainset, testset in kf.split(data):
    # train and test algorithm.
    algo.fit(trainingSet)
    predictions = algo.test(testset)
    accuracy.rmse(predictions, verbose=True)


# In[25]:


trainset = algo.trainset
print(algo.__class__.__name__)


# In[26]:


"""
Prediction on Test Data 
"""

result=[]
result1=[]
id=[]
user_id =[]
joke_id = []

for index, row in dt.iterrows():
    id.append(str((row['id'])) + '-' + str((row['user_id']))+'-'+str((row['joke_id'])))
    result1.append(algo.predict(row['user_id'], row['joke_id']).est)

result=pd.DataFrame({'id':pd.Series(id),'rating':pd.Series(result1) })
result[['id','user_id','joke_id']] = result['id'].str.split('-',expand=True)


# In[29]:


endResult = result.drop(['user_id','joke_id'],axis=1)
endResult.columns = ['id','Rating']
result.to_csv("sample_submission_5ms57N3.csv")

