# -*- coding: utf-8 -*-
"""
Created on Sat Jan 13 01:15:51 2018

Example:
    
from numpy import sin
x0 = newton(sin,2)
print "Solution is X = "+str(x0)

x0 = newton(_coupole,[2,2])
print "Solution is X = "+str(x0)

@author: rfetick
"""

from numpy import size, zeros, arange, array
from numpy.linalg import norm

def newton(funct,x0):
    """
    ### USAGE ###
     Newton's algorithm to find X such as f(X) = 0
     "X" is a vector of size N. Eventually N=1, so X is scalar
     "f" takes a vector of size N, and returns a scalar
    ### INPUTS ###
     funct - [function] The function to call
     X0    - [Vector] Initial guess on X
    ### OUTPUT ###
     Xzero - [Vector] Solution of f(X)=0
    """
    
    # INITIALIZE
    x = x0
    sX = size(x)
    dx = zeros(sX)
    idx = zeros(sX)
    h = 1e-10 #step to calculate derivative
    max_iter = 3000 #stop criteria: number of iterations
    dx_min = 1e-7 #stop criteria: x not evolving any more
    f_min = 1e-10 #stop criteria: f(x) close from 0 
    stop_criteria = 'max iteration reached'
    stop_bool = False
    
    # LOOP
    i = 0
    while i<max_iter and not stop_bool:
        f = funct(array(x))
        for iX in arange(sX):
            idx = 0*idx
            idx[iX] = h
            df = (funct(array(x)+array(idx))-f)/h
            dx[iX] = - f/df
        
        x2 = x + dx
        # If f(x) close from zero, stop algorithm
        if abs(f) < f_min:
            stop_bool = True
            stop_criteria = 'f(x) small enough'
        
        # If x is not moving, stop algorithm
        if norm(x-x2,ord=2) < dx_min:
            stop_bool = True
            stop_criteria = 'x is not moving any more'
            
        x = x2
        i = i+1
        
    # RESULTS
    print '[Iter='+str(i)+'] Stop criteria: '+stop_criteria
    return x


###############################################################################
#         Example function to use in newton(.)                                #
###############################################################################

def _coupole(x):
    return (x[0]-2.7)**2 + (x[1]-3.14159)**2