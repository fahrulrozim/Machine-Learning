
# coding: utf-8

# In[1]:


import pandas
import numpy as np
import matplotlib.pyplot as plt
import math
import csv


# In[2]:


#tampilan data awal
url ='C:/Users/HOME/Desktop/Data_iris.csv'
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
data = pandas.read_csv(url, names=names)


# In[3]:


print(data)


# In[45]:


data_iris = np.zeros((100,4))
class_fact = np.zeros((100))
def saveData():
    for i in range (100):
        for j in range(5):
            if j == 4:
                if data.iloc[i][j]=="Iris-setosa":
                    class_fact[i]=1
                else:
                    class_fact[i]=0
            else:
                data_iris[i][j] = data.iloc[i][j]
            


# In[127]:


saveData()
class_fact[0]


# In[97]:


tetha = [0.2, 0.6, 0.3, 0.4]
bias = 0.9
error = []
dTetha = []
dBias = 0.0
alpha = 0.1
#alpha = 0.8
prediksi = []


# In[101]:


#fungsi h(x,tetha,bias)
def h_function(data_iris, tetha, bias):
    value = 0.0
    for i in range(4):
        value += data_iris[0][i] * tetha[i]
    value += bias
    return value


# In[49]:


h_function(data_iris,tetha,bias)


# In[72]:


#fungsi sigmoid(h)
def sigmoid(h):
    return 1/(1+math.exp(-1.0*h_function(data_iris,tetha,bias)))


# In[73]:


h = h_function(data_iris, tetha, bias)
sigmoid(h)


# In[111]:


#Error
def error_function(prediction, fact):
    return (prediction - fact)**2


# In[130]:


#Delta Tetha
def delta_tetha(prediction, fact, data_iris):
    return 2*(prediction - fact)*(1 - fact) * fact * data_iris[j]


# In[86]:


#Update Tetha
def update_tetha(tetaI, deltaT):
        return tetaI - (alpha*deltaT)


# In[59]:


update_tetha(0)
array_baru


# In[115]:


#Delta Bias
def delta_bias(deltaBias):
    return 2*(prediction - fact) * (1 - fact) * fact


# In[61]:


delta_bias(0)


# In[114]:


#Update Bias
def update_bias(deltaBias):
    return bias -(alpha*deltaBias)


# In[63]:


update_bias(0)


# In[133]:


def start():
    for epoch in range(60):
        for i in range(100):
            bias = 0.9
            h = h_function(data_iris, tetha, bias)
            sig = sigmoid(h)
            prediction = class_fact[i]
            fact = sig
            err = error_function(sig, class_fact[i])
            error.append(err)
            prediksi.append(0 if err < 0.5 else 1)
            for j in range(4):
                dTetha[j] = delta_tetha(sigmoid, class_fact[i], data_iris[j])
                tetha[j] = update_tetha(tetha[j, dTetha[j]])
            dBias = delta_bias(sigmoid, class_fact[i])
            bias = update_bias(dbias)


# In[ ]:


start()
plt.plot(error)
plt.show

