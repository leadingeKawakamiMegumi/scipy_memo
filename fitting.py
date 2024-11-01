import numpy as np
import matplotlib.pyplot as plt
import matplotlib


#sigmoid--------------------------------------
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

#y=0.5---> x=b/a


#gauss--------------------------
from scipy.stats import mstats,norm
from scipy.optimize import curve_fit


def func(x, a, mu, sigma):
    return a*np.exp(-(x-mu)**2/(2*sigma**2))


#histgram
#param_ini = [20,0,1]  #a,mean,sigma
#p0 = param_ini
def gauss_fit_hist(data, p0, nn=40):
    plt.hist(data,nn,alpha=0.5)
    hist, bins = np.histogram(data,nn)
    bins=bins[:-1]
    popt, pcov = curve_fit(func, bins, hist, p0=p0) #pcov:共分散
    print("Intensity:",popt[0], " mean:",popt[1],"standard deviation:",popt[2])
    fitting = func(bins, popt[0],popt[1],popt[2])
    plt.plot(bins,fitting,alpha=0.5)
    plt.show()
    return(popt)


#dataそのまま
def gauss_fit(data, x,p0):
    '''
    data:観測値
    x:x軸
    p0:初期値
    '''
    popt, pcov = curve_fit(func, x, data, p0=p0) #pcov:共分散
    print("Intensity:",popt[0], " mean:",popt[1],"standard deviation:",popt[2])
    fitting = func(x, popt[0],popt[1],popt[2])
#     plt.plot(nm,fitting,alpha=0.5)
#     plt.show()
    return(popt,fitting)


#面積（ガウス分布関数の積分）
#k = popt[0]*(2*np.pi)**0.5*popt[2]
#ガウスで非対称な場合　JohnsonSU分布
#https://the-silkworms.com/ds_stat_johnsons-su_basic/160/
