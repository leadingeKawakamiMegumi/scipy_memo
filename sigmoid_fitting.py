import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit  


def sigmoid(X,a,b):
    Y1 =1+ np.exp(b - a*X)
    Y = 1 / Y1
    return Y


def inverse_sigmoid(y,a,b):
    '''logit function'''
    x = (b + np.log(y/(1-y)))/a
    return(x)


def sigmoid_fit(x_observed,y_observed,param_ini=(1.5,20),param_bounds= ([0,0],[np.inf,np.inf])): #min=0,max=1
    popt, pcov = curve_fit(sigmoid,x_observed,y_observed,p0 = param_ini,bounds=param_bounds) # poptは最適推定値、pcovは共分散
    y_fit = sigmoid(x_observed,popt[0],popt[1])
    plt.scatter(x_observed,y_observed,color = "gray")
    plt.plot(x_observed,y_fit)
    return(popt[0],popt[1],y_fit)
