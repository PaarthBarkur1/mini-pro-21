#!/usr/bin/env python
# coding: utf-8

# In[11]:


# this is naive bayes
# split is = 0 by default but if a test dataset is given it has the value of the csv
import pandas as pd
import numpy as np  #this is a fuction that just takes input and gibes out clf as of now takes no parameters
import pickle
import requests

def output(model1,var_smoothing1,n_neighbors1,leaf_size1,max_depth1,min_samples_split1,n_estimators1,random_state1,max_leaf_nodes1):
    model=model1 #insert the suitable choice here
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



        clf=KNeighborsClassifier(n_neighbors=n_neighbors1,leaf_size=leaf_size1)
        clf.fit(X_train,Y_train.values.ravel())
    
        return clf



    #this is decision tress
    def decisiontrees(max_depth=None,min_samples_split=2):
        from sklearn.tree import DecisionTreeClassifier



        clf=DecisionTreeClassifier(max_depth=max_depth,min_samples_split=int(min_samples_split))
        clf.fit(X_train,Y_train)
        

        return clf


    #regression begins
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


    if model == 'naivebayes':
        
        clf=naives(var_smoothing=var_smoothing1)
        pickle.dump(clf,open('model.pkl','wb'))
        modelf=pickle.load(open('model.pkl','rb'))
        
    elif model=='decisiontrees':
        
        clf=decisiontrees(max_depth=max_depth1,min_samples_split=min_samples_split1)
        pickle.dump(clf,open('model.pkl','wb'))
        modelf=pickle.load(open('model.pkl','rb'))
        
    elif model=='knn':
        
        clf=knn(n_neighbors=n_neighbors1,leaf_size=leaf_size1)
        pickle.dump(clf,open('model.pkl','wb'))
        modelf=pickle.load(open('model.pkl','rb')) 
        
    elif model=='randomforest':
        
        clf=randomforest(n_estimators=400 ,max_leaf_nodes=max_leaf_nodes1,min_samples_split=min_samples_split1) #as of now no parameters are passed
        pickle.dump(clf,open('model.pkl','wb'))
        modelf=pickle.load(open('model.pkl','rb'))
        
    elif model == 'linearreg':
        
        clf=linearreg()
        pickle.dump(clf,open('model.pkl','wb'))
        modelf=pickle.load(open('model.pkl','rb'))
        
    elif model=='decisiontreereg':
        
        clf=decisiontreereg(max_depth=max_depth1,random_state=random_state1,min_samples_split=min_samples_split1)
        pickle.dump(clf,open('model.pkl','wb'))
        modelf=pickle.load(open('model.pkl','rb'))
        
    elif model=='knnreg':
        
        clf=knnreg(n_neighbors=n_neighbors1,leaf_size=leaf_size1)
        pickle.dump(clf,open('model.pkl','wb'))
        modelf=pickle.load(open('model.pkl','rb')) 
        
    elif model=='randomforestreg':
        
        clf=randomforestreg(n_estimators=n_neighbors1, max_leaf_nodes=max_leaf_nodes1) #as of now no parameters are passed
        pickle.dump(clf,open('model.pkl','wb'))
        modelf=pickle.load(open('model.pkl','rb'))

    