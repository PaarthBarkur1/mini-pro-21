#!/usr/bin/env python
# coding: utf-8

# In[11]:


# this is naive bayes
# split is = 0 by default but if a test dataset is given it has the value of the csv
model='knn' #insert the suitable choice here
import pandas as pd
import numpy as np  #this is a fuction that just takes input and gibes out clf as of now takes no parameters
import pickle
X_train=pd.read_csv('dftraincls.csv')
Y_train=pd.read_csv('ytraincls.csv')
def naives(var_smoothing=1e-9):
    from sklearn.naive_bayes import GaussianNB


    clf=GaussianNB()
    clf.fit(X_train,Y_train.values.ravel())

    return clf



def randomforest(n_estimators=500,max_leaf_nodes=None,min_samples_split=2):
    from sklearn.ensemble import RandomForestClassifier


    clf=RandomForestClassifier(n_estimators=n_estimators,max_leaf_nodes=max_leaf_nodes,min_samples_split=min_samples_split,n_jobs = -1)
    clf.fit(X_train,Y_train.values.ravel())

    return clf



def knn(n_neighbors = 5 ,leaf_size = 30):
    from sklearn.neighbors import KNeighborsClassifier 



    clf=KNeighborsClassifier(n_neighbors=n_neighbors,leaf_size=leaf_size)
    clf.fit(X_train,Y_train.values.ravel())
   
    return clf



#this is decision tress
def decisiontrees(max_depth=None,min_samples_split=2):
    from sklearn.tree import DecisionTreeClassifier



    clf=DecisionTreeClassifier(max_depth=max_depth,min_samples_split=int(min_samples_split))
    clf.fit(X_train,Y_train)
    

    return clf




if model == 'naivebayes':
    
    clf=naives(var_smoothing=1)
    pickle.dump(clf,open('modelnb.pkl','wb'))
    modelnb=pickle.load(open('modelnb.pkl','rb'))
    
elif model=='decisiontrees':
    
    clf=decisiontrees(max_depth=6)
    pickle.dump(clf,open('modeldtc.pkl','wb'))
    modeldtc=pickle.load(open('modeldtc.pkl','rb'))
    
elif model=='knn':
    
    clf=knn(n_neighbors=6,leaf_size=10)
    pickle.dump(clf,open('modelknn.pkl','wb'))
    modelknn=pickle.load(open('modelknn.pkl','rb')) 
    
elif model=='randomforest':
    
    clf=randomforest(n_estimators=400 ) #as of now no parameters are passed
    pickle.dump(clf,open('modelrfc.pkl','wb'))
    modelrfc=pickle.load(open('modelrfc.pkl','rb'))
    


# In[18]:


x=253
print(modelknn.predict(X_train.iloc[[x]]))
Y_train.iloc[[x]]


# In[ ]:




