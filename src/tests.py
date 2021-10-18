'''
Created on 26.4.2020

@author: sofievm
'''

import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model


def check_regr_ortogonalization():
    A = np.array([2.0, 3.14])
    B = np.array([15, 27.2])
    t = np.array(range(100), dtype=np.float32)
    
    x = np.zeros(shape=(100, 1))
    y = np.zeros(shape=(100))
    
    x[:,0] = (A[0] + np.sin(0.3 * t)) * t + B[0]
    y[:] = (A[1] + np.sin(-0.3 * t)) * t[:] + B[1]
    
    
    print(x.shape, y.shape)
    clf = linear_model.LinearRegression()
    clf.fit(x, y)   # x_train, y_train
    yPr = clf.predict(x)
    resid = yPr - y
    print('Coefficients:', clf.coef_)
    print('intercept', clf.intercept_)
    print('res_mean', np.mean(resid), 'corr(x,resid)', np.corrcoef(x.T,resid.T)[0], 
          'corr(resid,y)', np.corrcoef(y.T,resid.T)[0])
        
    fig = plt.figure()
    ax = fig.subplots()
    ax.plot(t,x,label='x')
    ax.plot(t,y,label='y')
    ax.plot(t,yPr,label='linregr(y=a*x+b)')
    ax.plot(t,resid,label='residual')
    ax.legend()
    fig.savefig('d:\\tmp\\try_regr.png', dpi=200)
    plt.clf()
    plt.close()
    
    
    print(np.corrcoef(x.T,resid.T))
    fig = plt.figure()
    ax = fig.subplots()
    ax.scatter(x, resid, label='x')
    fig.savefig('d:\\tmp\\try_regr_scatter.png', dpi=200)
    plt.clf()
    plt.close()
    


if __name__ == '__main__':
    check_regr_ortogonalization()