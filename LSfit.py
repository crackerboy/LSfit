# -*- coding: utf-8 -*-
"""
Created on Tue Jan 02 17:57:41 2018
Updated on Sat Jan 13 13:01:00 2018

WEIGHTS = 1/sigma^2 for a homogeneous gaussian noise
WEIGHTS = 1/data for a Poisson-like noise

TO-DO
- Create a LSfit2D(.)
- Define more precisely stopping criteria for algorithm
- Check Levenberg-Marquardt regularization
- Check the WEIGHTED least-square convergence

@author: rfetick
"""

# arange might be used as a string in eval('arange(.)')
# so I let it, whatever the warning says
from numpy import size, zeros, arange, array, dot, diag
from scipy.linalg import inv


def LSfit(funct,data,X,param0, weights=-1, quiet=False, LM=False):
    """
    ### USAGE ###
     Least-square minimization algorithm.
     Parametric least-square fitting.
     Minimises Xhi2 = sum(weights*(f(X,param) - data)^2)
     where the sum(.) runs over the coordinates X
     "data" is the noisy observation
     "f(X,param)" is your model, set by the vector of parameters "param" and evaluated on coordinates X
    ### INPUTS ###
     funct   - [Function] The function to call
     data    - [Vector] Noisy data
     X       - [Vector] Coordinates where to evaluate the function
     param0  - [Vector] First estimation of parameters
    ### OPTIONAL INPUTS ###
     weights - [Vector] Weights (1/noise^2) on each coordinate X
                  Default : no weighting (weights=1)
                  weights = 1/sigma^2 for a gaussian noise
                  weights = 1/data for a Poisson-like noise
     quiet   - [Boolean] Don't print status of algorithm
                  Default : quiet = False
     LM      - [Boolean] Levenberg-Marquardt algorithm
                  Default : LM = False
    ### OUTPUT ###
     param   - [Vector] Best parameters for least-square fitting
    """
    # INITIALIZATION
    sX = size(X)
    sP = size(param0)
    J = zeros((sX,sP)) # Jacobian
    h = 1e-9 #step to compute Jacobian
    param = param0.copy()
    
    # STOPPING CONDITIONS
    J_min = 1e-5*sP*sP # stopping criteria based on small Jacobian
    dp_min = 1e-7 # stopping criteria based on small variation of parameters
    max_iter = 3000 # stopping criteria based on max number of iteration
    stop_loop = False # boolean for stopping the loop
    stop_trace = "Maximum iteration reached (iter="+str(max_iter)+")"
    
    # WEIGHTS
    if size(weights)==1 and weights<=0:
        weights = zeros(sX)+1
    elif size(weights)>1 and size(weights)!=sX:
        raise ValueError("WEIGHTS should be a scalar or same size as X")
    
    # LEVENBERG-MARQUARDT
    mu = 0
    if LM:
        mu = 1
    
    # LOOP
    i = 0
    while (i < max_iter) and not stop_loop:
        
        f = funct(array(X),array(param))
        
        ## Iterate over each parameter to compute derivative
        for ip in arange(sP):
            dparam = zeros(sP)
            dparam[ip] = h
            
            J[:,ip] = weights*(funct(array(X),array(param)+array(dparam))-f)/h
        
        ## Compute dparam = -{(transpose(J)*J)^(-1)}*transpose(J)*(func-data)
        JTJ = dot(J.T,J)
        dparam = - dot(dot(inv(JTJ+mu*diag(JTJ.diagonal())),J.T),weights*(f-data))
        param = param + dparam
        
        ## Xhi square
        Xhi2 = sum(weights*(f-data)**2)
        
        ## Print Xhi square
        if not quiet:
            Xhi2PRINT = "[Iter="+str(i)+"] Xhi2 = "+str(Xhi2)
            if not LM:
                print Xhi2PRINT
            else:
                print Xhi2PRINT + " [mu="+str(mu)+"]"
                
        ## Levenberg-Marquardt update for mu
        if LM:
            f = funct(X,param)
            Xhi2_new = sum(weights*(f-data)**2)
            if Xhi2_new > Xhi2:
                mu = 10*mu
            else:
                mu = 0.1*mu
        
        ##Stop loop based on small variation of parameter
        if sum(abs(dparam)) < dp_min*sum(abs(param)):
            stop_loop = True
            stop_trace = "Parameter not evolving enough at iter="+str(i)
        
        ## Stop loop based on small Jacobian
        if abs(J).sum() < J_min:
            stop_loop = True
            stop_trace = "Jacobian small enough at iter="+str(i)
        
        ## Increment
        i = i+1  
    
    if not quiet:
        print " "
        print "Stopping condition: "+stop_trace
        print " "
    
    return param