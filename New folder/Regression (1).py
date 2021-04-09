#!/usr/bin/env python
# coding: utf-8

# In[43]:


# regression
# split is = 0 by default but if a test dataset is given it has the value of the csv
model='decisiontreereg'
import pandas as pd
import numpy as np  #this is a fuction that just takes input and gibes out clf as of now takes no parameters
import pickle
X_train=pd.read_csv('dftraincls.csv')
Y_train=pd.read_csv('ytraincls.csv')

def randomforestreg(n_estimators=500,max_leaf_nodes=None):
    from sklearn.ensemble import RandomForestRegressor

    clf=RandomForestRegressor(n_estimators=n_estimators,max_leaf_nodes=max_leaf_nodes,n_jobs = -1)
    clf.fit(X_train,Y_train.values.ravel())

    return clf



def knnreg(n_neighbors = 5 ,leaf_size = 30):
    from sklearn.neighbors import KNeighborsRegressor 

    clf=KNeighborsRegressor(n_neighbors=n_neighbors,leaf_size=leaf_size)
    clf.fit(X_train,Y_train.values.ravel())
   
    return clf



#this 
def linearreg():
    from sklearn.linear_model import LinearRegression


    clf=LinearRegression()
    clf.fit(X_train,Y_train)
    

    return clf

def decisiontreereg( max_depth=None,random_state=42,min_samples_split=2):
    from sklearn.tree import DecisionTreeRegressor

    clf=DecisionTreeRegressor(max_depth=max_depth,random_state=random_state,min_samples_split=int(min_samples_split))
    clf.fit(X_train,Y_train)
    

    return clf



if model == 'linearreg':
    
    clf=linearreg()
    pickle.dump(clf,open('modellr.pkl','wb'))
    modellr=pickle.load(open('modellr.pkl','rb'))
    
elif model=='decisiontreereg':
    
    clf=decisiontreereg(max_depth=4)
    pickle.dump(clf,open('modeldtcreg.pkl','wb'))
    modeldtcreg=pickle.load(open('modeldtcreg.pkl','rb'))
    
elif model=='knnreg':
    
    clf=knnreg(n_neighbors=6,leaf_size=10)
    pickle.dump(clf,open('modelknnreg.pkl','wb'))
    modelknnreg=pickle.load(open('modelknnreg.pkl','rb')) 
    
elif model=='randomforestreg':
    
    clf=randomforestreg(n_estimators=400 ) #as of now no parameters are passed
    pickle.dump(clf,open('modelrfcreg.pkl','wb'))
    modelrfcreg=pickle.load(open('modelrfcreg.pkl','rb'))
    

    


# In[54]:


x=655
print(modeldtcreg.predict(X_train.iloc[[x]]))
Y_train.iloc[[x]]


# In[29]:





# In[ ]:




