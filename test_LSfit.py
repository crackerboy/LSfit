# -*- coding: utf-8 -*-
"""
Created on Thu Jan 04 17:51:47 2018

Test the LSfit method
Be sure to import the 'gauss' method to run this test

@author: rfetick
"""

# IMPORT
from numpy import linspace, array, floor
from numpy.random import rand, poisson
from pylab import plot, legend, show



###############################################################################
###         Gaussian
###############################################################################

print "-------------- Gaussian --------------"

Npoints = 50

# DEFINE NOISY DATA
X = linspace(0,10,num=Npoints)
paramTrue = array([1.,5.,1.,5.])
Ytrue = gauss(X,paramTrue)
Ynoisy = Ytrue + 2*(rand(Npoints)-.5)

# MINIMIZATION
param0 = array([1.5,4.,2.,5.5])
Ystart = gauss(X,param0)
param = LSfit(gauss,Ynoisy,X,param0,LM=True)
Yfit = gauss(X,param)

# SHOW RESULTS
print "Param      : [bck,amp,sig,mid]"
print "Param true : "+str(paramTrue)
print "Param start: "+str(param0)
print "Param fit  : "+str(floor(param*100)/100.0)

plot(X,Ynoisy,'orange',linewidth=2)
plot(X,Ytrue,'b',linewidth=3)
plot(X,Ystart,'g',linewidth=2)
plot(X,Yfit,'r',linewidth=2)
legend(("Noisy data","True curve","Init fitting","LSfit solution"))
show()



###############################################################################
###         MOFFAT
###############################################################################

print "-------------- Moffat --------------"

# DEFINE NOISY DATA
paramTrue = array([1.,5.,1.,5.,2.])
Ytrue = moffat(X,paramTrue)
Ynoisy = Ytrue + 2*(rand(Npoints)-.5)

# MINIMIZATION
param0 = array([1.5,4.,2.,5.5,3.])
Ystart = moffat(X,param0)
param = LSfit(moffat,Ynoisy,X,param0,LM=True,debug=500)
Yfit = moffat(X,param)

# SHOW RESULTS
print "Param      : [bck,amp,sig,mid,pow]"
print "Param true : "+str(paramTrue)
print "Param start: "+str(param0)
print "Param fit  : "+str(floor(param*100)/500.0)

plot(X,Ynoisy,'orange',linewidth=2)
plot(X,Ytrue,'b',linewidth=3)
plot(X,Ystart,'g',linewidth=2)
plot(X,Yfit,'r',linewidth=2)
legend(("Noisy data","True curve","Init fitting","LSfit solution"))
show()


###############################################################################
###                  MOFFAT using class LSparam
###     Check class LSparam from LSfit.py for more information
###############################################################################

# I introduce on purpose bad constraints on the parameters to show you the effects
# However this may lead to errors of convergence

print "-------------- Moffat LSparam --------------"

# DEFINE NOISY DATA
paramTrue = array([1.,5.,1.,5.,2.])
Ytrue = moffat(X,paramTrue)
Ynoisy = Ytrue + 2*(rand(Npoints)-.5)

# MINIMIZATION
param0 = LSparam([2.,4.,2.,5.5,3.])
param0.fixed = [False,False,True,False,True]
#param0.set_bound_down(2)
Ystart = moffat(X,param0.value)
param = LSfit(moffat,Ynoisy,X,param0,LM=True,debug=10)
Yfit = moffat(X,param)

# SHOW RESULTS
print "Param      : [bck,amp,sig,mid,pow]"
print "Param true : "+str(paramTrue)
print "Param start: "+str(param0.value)
print "Param fixed: "+str(param0.fixed)
print "Param fit  : "+str(floor(param*100)/100.0)

plot(X,Ynoisy,'orange',linewidth=2)
plot(X,Ytrue,'b',linewidth=3)
plot(X,Ystart,'g',linewidth=2)
plot(X,Yfit,'r',linewidth=2)
legend(("Noisy data","True curve","Init fitting","LSfit solution"))
show()