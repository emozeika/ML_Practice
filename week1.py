 89# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:48:44 2019

@author: mozei
"""

"""Machine Learning Andrew Ng Homework 1"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

"""Problem 1: Create a 5 by 5 identity matrix"""

diag_mat = np.diag(np.ones(5))
print(diag_mat)


"""Problem 2: Univariate Linear Regression"""

###make a scatter plot of the data
data1 = pd.read_csv('C:\\Users\mozei\Desktop\Andrew_NG_ML\ml1\ml1\ex1\ex1data1.txt', header=None)
X = data1.iloc[:,0]
y = data1.iloc[:,1]
m = len(X)

plt.scatter(X,y)
plt.xlabel('Population of city in 10,000s')
plt.ylabel('Profit in $10,000')
plt.show()

###Gradient descent

#turn the x and y into vectors; x needs a 1s vector for the intercept
X = X[:,np.newaxis]
X = np.hstack((np.ones((m, 1)), X))
y = y[:,np.newaxis]

#set inital parameters, iterations 
theta = np.zeros((2,1))
alpha = 0.01
iterations = 1500

#cost function equation
def ComputeCost(X, y, theta):
    #X: matrix to store values of our features
    #y: vector for our label values
    #thetas: coefficents of our X values    
    m = len(X)
    J = (np.sum((np.dot(X, theta) - y)**2))/(2*m)
    return J

ComputeCost(X, y, theta)
    

#compute the values from gradient descent
def GradientDescent(X, y, theta, alpha, num_iterations):
    #X: matrix to store values of our features
    #y: vector for our label values
    #thetas: coefficents of our X values
    #alpha: learning rate
    #num_iterations: number of iterations to run gradient descent

    m = len(y)
    for i in range(iterations):
        theta = theta - (alpha/m)*np.dot(X.T,(np.dot(X, theta)-y))
    return theta


new_thetas = GradientDescent(X, y, theta, alpha, iterations)
print(new_thetas)
ComputeCost(X, y, new_thetas)

#visualize the linear regression

plt.scatter(X[:,1], y)
plt.plot(X[:,1], np.dot(X,new_thetas), color = 'red')
plt.xlabel('Population of city in 10,000s')
plt.ylabel('Profit in $10,000')
plt.title('Scatter plot with Linear Fit')
plt.show()

""" Problem 3 Multivariate Linear Regression"""

data2 = pd.read_csv('C:\\Users\mozei\Desktop\ml1\ml1\ex1\ex1data2.txt', header=None)
data2.head()

m = len(X)
X = data2.iloc[:,:2]
y = data2.iloc[:,2]

np.mean(X)

#function to normalize features. subtract mean from feature value and sclae it by dividing by std dev
def FeatureNormalization(df):
    #df:  dataset to run feature normalization on

    df = (df - np.mean(df))/np.std(df)    
    return df
        
X = FeatureNormalization(X)


#computing the cost function
X = np.hstack((np.ones((m, 1)), X))
y = y[:,np.newaxis]


thetas = np.zeros((3,1))


def ComputeCostMulti(X, y, theta):
    #X: matrix to store values of our features
    #y: vector for our label values
    #thetas: coefficents of our X values  
    
    m = len(y)
    J_theta = (1/(2*m))*(np.dot((np.dot(X,theta)-y).T,np.dot(X,theta)-y))
    
    return J_theta
    
ComputeCostMulti(X,y,thetas)


#computing the gradient descent
def GradientDescentMulti(X,y,theta,alpha,num_iters):
    
    Cost = []
    Theta = []
    for i in range(num_iters):
        theta = theta - (alpha/m)*np.dot(X.T,(np.dot(X, theta)-y))
        cost = np.asscalar(ComputeCostMulti(X,y,theta))
        Cost.append(cost)
        Theta.append(theta)
        
    return(Theta, np.asarray(Cost))


#visualize the cost function through Gradient Descent
gd_thetas, gd_cost = GradientDescentMulti(X,y,thetas,0.3, 50)



plt.plot(range(len(gd_cost)), gd_cost)
plt.xlabel('Number of Iterations')
plt.ylabel('Cost Function Value')
plt.show()


#optimal thetas
print(gd_thetas[-1])
print(gd_cost[-1])

