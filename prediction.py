
# coding: utf-8

# ## About the Data
# 
# https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
# Note: This data is availible online. Please do not download or use that data because that would be cheating
# 
# Your job is to classify a tumor as benign (0) or malignant (1)
# 
# 
# ### Imports

# In[21]:


##tools
from numpy import genfromtxt
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

##a couple of models
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# ### Loading the Data 

# In[3]:


data = genfromtxt('wdbc_data.csv', delimiter=',', skip_header=1, dtype=float) ## Split this into test and train

x_pred = genfromtxt('wdbc_pred.csv', delimiter=',', skip_header=1, dtype=float) ## Data for when you submit your prediction


# In[15]:


x = train[:,0:30] ## columns containing the attributes
y = train[:,30] ## class column ( 0 or 1)

x_train, x_test, y_train, y_test = train_test_split(x, y)

print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)


# In[ ]:


## For submission, change "test" to your prediction
np.savetxt("pred.txt", test, fmt='%.18e', delimiter=',', newline='\n', header='', footer='', comments='# ')

