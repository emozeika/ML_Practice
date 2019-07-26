# -*- coding: utf-8 -*-
"""
Created on Mon Jul 22 18:02:01 2019

@author: mozei
"""

"""Machine Learning Andrew Ng Homework 2"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

"""Part 1: Logisitic Regression"""

data1 = pd.read_csv('C:\\Users\mozei\Desktop\Andrew_NG_ML\ml2\ex2\ex2data1.txt', header=None)
data1.head()

#Problem 1: Make a function to visualize the data

def PlotData(df):
    #df; datframe consisting of 2 exams scores and indicator variavle of applicants addimtance
    
    plt.scatter(df.iloc[:,0], df.iloc[:,1], c=df.iloc[:,2])
    plt.xlabel('Exam Score 1')
    plt.ylabel('Exam Score 2')
    plt.show()
    
PlotData(data1)


#Problem 2: Create a sigmoid funciton

def Sigmoid(z):
    g_z = 1/(1 + np.exp(-z))
    return g_z

m = len(y)
X = data1.iloc[:, :2]
X = np.hstack((np.ones((m,1)), X))
y = np.array(data1.iloc[:, 2]).reshape(-1,1)
theta = np.array([0.0, 0.0, 0.0])


#Problem 3: Crate cost function and gradient for logisitic regression

def CostFunction(theta, X, y):
    #X: feature values of your data
    #y: classifier label of your data
    #theta: inital parameter guess
    
    
    m = len(y)
    g_theta = Sigmoid(X.dot(theta))
    j_theta = (-1/m)*(np.log(g_theta).T.dot(y) + np.log(1-g_theta).T.dot(1-y))
    return j_theta[0]

def Gradient(theta, X, y):
    #X: feature values of your data
    #y: classifier label of your data
    #theta: inital parameter guess
    
    
    m= len(y)
    g_theta = Sigmoid(X.dot(theta.reshape(-1,1)))
    gradient = (1/m)* X.T.dot(g_theta-y)
    
    return gradient.flatten()



t = CostFunction(theta, X, y)
g = Gradient(theta, X, y)


#Problem 5: Learning parameters

res = minimize(CostFunction, theta, args = (X, y), method = None,jac = Gradient, options={'maxiter':400})

opt_theta = res['x']


#Problem 6: Predict wether a future score will be admitted or not

def Predict(theta, X, threshold = 0.50):
    #optimal parameter values
    #X: vector of 1 and 2 scores of exams
    #threshold for positive result
    
    prob = Sigmoid(X.dot(theta))
    if prob >= threshold:
        return 1
    else: 
        return 0


Predict(opt_theta, np.array([1, 45, 85]))


"""Part 2: Regularized Logisitic Regression"""

#Problem 1:Visualize the data
data2 = pd.read_csv('C:\\Users\mozei\Desktop\Andrew_NG_ML\ml2\ex2\ex2data2.txt', header=None)
data2.head()

PlotData(data2)

#Problem 2: Create more features for the data

def MapFeature(df):
    #df: dataframe of data
    
    
    X = df.iloc[:,:2]
    m = df.shape[0]
    
    #6 degrees, we have first degree already
    for i in range(2,7):
        #terms per degree range ie) degree 2 has 3 terms
        for j in range(0, i+1):
            term = X.iloc[:,0]**(i-j) * X.iloc[:,1]**(j)
            X = pd.concat([X,term], axis = 1)
            
    #add the intercept
    X = np.hstack((np.ones((m,1)),X))
    
    return X
    
X = np.hstack((np.ones((data2.shape[0],1)),data2.iloc[:,:2]))
y = np.array(data2.iloc[:, 2]).reshape(-1,1)
int_theta = np.zeros((28,1))

X = MapFeature(data2)
#Problem 3: Compute Regularized Cost Function

def ComputeRegCost(theta, X, y, lamb=0):
    #theta: coeeficent values
    #X: feature values
    #y: label values
    #lamb: penalty parameter
    
    m = len(y)
    g_theta = Sigmoid(X.dot(theta))
    j_theta = (-1/m)*(np.log(g_theta).T.dot(y) + np.log(1-g_theta).T.dot(1-y)) + np.sum(theta[1:]**2)*(lamb/(2*m))
    
    return j_theta
    

def GradientReg(theta, X, y, lamb=0):
    #theta: coeeficent values
    #X: feature values
    #y: label values
    #lamb: penalty parameter
    
    m= len(y)
    g_theta = Sigmoid(X.dot(theta.reshape(-1,1)))
    gradient = (1/m)* X.T.dot(g_theta-y) + np.r_[[[0]],(lamb/m)*theta[1:].reshape(-1,1)]
    
    return gradient.flatten()
    

res = minimize(ComputeRegCost, int_theta, args = (X, y, 0), method = None,jac = GradientReg, options={'maxiter':400})

res['sol']