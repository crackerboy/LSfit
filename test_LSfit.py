# -*- coding: utf-8 -*-
"""
Created on Thu Jan 04 17:51:47 2018

Test the LSfit method
Be sure to import the 'gauss' method to run this test

@author: rfetick
"""

# IMPORT
from numpy import linspace, array
from numpy.random import rand
from pylab import plot, legend

# DEFINE NOISY DATA
X = linspace(0,10,num=100)
paramTrue = array([1.,5.,1.,5.])
Ytrue = gauss(X,paramTrue)
Ynoisy = Ytrue + 2*(rand(100)-.5)

# MINIMIZATION
param0 = array([1.5,4.,2.,5.5])
Ystart = gauss(X,param0)
param = LSfit('gauss',Ynoisy,X,param0,LM=True)
Yfit = gauss(X,param)

# SHOW RESULTS
print "Param      : [bck,amp,sig,mid]"
print "Param true : "+str(paramTrue)
print "Param start: "+str(param0)
print "Param fit  : "+str(param)
plot(X,Ytrue,'b',linewidth=3)
plot(X,Ynoisy,'orange',linewidth=2)
plot(X,Ystart,'g',linewidth=2)
plot(X,Yfit,'r',linewidth=2)
legend(("True","Noisy","Start","Fit"))
