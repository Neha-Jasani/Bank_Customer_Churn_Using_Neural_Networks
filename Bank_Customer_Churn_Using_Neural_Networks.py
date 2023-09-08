#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('Churn_Modelling.csv')


# In[3]:


df.head()


# In[4]:


df.shape


# In[5]:


df.describe().transpose()


# In[6]:


df.info()


# In[7]:


sns.countplot(data = df, x='IsActiveMember')


# In[8]:


sns.countplot(data = df, x='Exited')


# In[9]:


# Intializing the 'X' and 'Y'
X = df.iloc[:,3:-1].values
y=df.iloc[:,-1].values


# In[10]:


X


# In[11]:


y


# #### Encoding categorical data

# In[12]:


# 1. label encoding the gender column


# In[13]:


from sklearn.preprocessing import LabelEncoder


# In[14]:


le = LabelEncoder()


# In[15]:


X[:,2] = le.fit_transform(X[:,2])


# In[16]:


print(X)


# In[17]:


# 2. Onehot encoding the Geography column


# In[18]:


from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# In[19]:


ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
X = np.array(ct.fit_transform(X))


# In[20]:


print(X)


# ##### Feature Scaling

# In[21]:


from sklearn.preprocessing import StandardScaler


# In[22]:


sc = StandardScaler()


# In[23]:


X = sc.fit_transform(X)


# In[24]:


print(X)


# In[25]:


# Spliting dataset into training set ans test set


# In[26]:


from sklearn.model_selection import train_test_split


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# In[ ]:


import keras
from keras.layers import Dense
from keras.models import Sequential


# In[ ]:


model = Sequential([Dense(units=6,activation='relu'),    # Adding Input layer and the first hidden layer
                    Dense(units=6,activation='relu'),    # Adding Second hidden layer
                    Dense(units=1,activation='sigmoid'), # Adding output layer
                    ])


# In[ ]:


# Compiling the ANN
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[ ]:


# Training the ANN on the Training set
ANN_model = model.fit(X_train, y_train, batch_size=32, epochs=100)


# In[ ]:


plt.plot(ANN_model.history['accuracy'], label='Accuracy', color='blue')
plt.plot(ANN_model.history['loss'], label='Loss', color='red')
plt.legend()


# In[ ]:


ann_pred = model.predict(X_test)


# In[ ]:


print(ann_pred)


# In[ ]:


y_pred = [1 if y>=0.5 else 0 for y in ann_pred]


# In[ ]:


print(y_pred)


# In[ ]:


from sklearn.metrics import classification_report
from sklearn import metrics


# In[ ]:


print(classification_report(y_test, y_pred))


# In[ ]:




