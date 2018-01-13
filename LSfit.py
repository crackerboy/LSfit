# -*- coding: utf-8 -*-
"""
Created on Tue Jan 02 17:57:41 2018
Updated on Sat Jan 13 13:01:00 2018
    Removed these ugly eval(.)

INFORMATIONS
Errors might occur due to bad matrix conditioning.
Please be careful with that

TO-DO
- Create a LSfit2D(.)
- Check Levenberg-Marquardt regularization
- Check the WEIGHTED least-square convergence
- Errors mights occur sometimes due to bad matrix conditioning
    Improve matrix conditioning when (or before) a LinAlgError is raised
    
Licence GNU-GPL v3.0

@author: rfetick
"""

# arange might be used as a string in eval('arange(.)')
# so I let it, whatever the warning says
from numpy import size, zeros, arange, array, dot, diag, floor, log10, inf, NaN
from numpy.linalg import cond, eig
from scipy.linalg import inv, LinAlgError


def LSfit(funct,data,X,param0, weights=-1, quiet=False, LM=False, debug=False):
    """
    
    #####  USAGE  #####
     Least-square minimization algorithm.
     Parametric least-square fitting.
     
     Minimises Xhi2 = sum(weights*(f(X,param) - data)^2)
     where the sum(.) runs over the coordinates X
     minimization is performed over the unknown "param"
     "data" is the noisy observation
     "f(X,param)" is your model, set by the vector of parameters "param" and evaluated on coordinates X
    
    #####  INPUTS  #####
     funct   - [Function] The function to call
     data    - [Vector] Noisy data
     X       - [Vector] Coordinates where to evaluate the function
     param0  - [Vector] First estimation of parameters
    
    ##### OPTIONAL INPUTS #####
     weights - [Vector] Weights (1/noise^2) on each coordinate X
                  Default : no weighting (weights=1)
                  weights = 1/sigma^2 for a gaussian noise
                  weights = 1/data for a Poisson-like noise
     quiet   - [Boolean] Don't print status of algorithm
                  Default : quiet = False
     LM      - [Boolean] Levenberg-Marquardt algorithm
                  Default : LM = False
    
    #####  OUTPUT  #####
     param   - [Vector] Best parameters for least-square fitting
    """
    # INITIALIZATION
    sX = size(X)
    sP = size(param0)
    J = zeros((sX,sP)) # Jacobian
    h = 1e-9 #step to compute Jacobian
    param = param0.copy()
    
    dparam = zeros(sP)
    
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
    
    # Print some information
    if not quiet:
        f = funct(array(X),array(param))
        Xhi2 = sum(weights*(f-data)**2)
        print "[Iter=0] Xhi2 = "+str(Xhi2)
    
    
    # LOOP
    
    iteration = 0
    
    while (iteration < max_iter) and not stop_loop:
        
        f = funct(array(X),array(param))
        
        ## Iterate over each parameter to compute derivative
        for ip in arange(sP):
            dparam = 0*dparam
            dparam[ip] = h
            
            J[:,ip] = weights*(funct(array(X),array(param)+array(dparam))-f)/h
        
        ## Compute dparam = -{(transpose(J)*J)^(-1)}*transpose(J)*(func-data)
        JTJ = dot(J.T,J)
        try:
            dparam = - dot(dot(inv(JTJ+mu*diag(JTJ.diagonal())),J.T),weights*(f-data))
        except LinAlgError as exception_message:
            # These errors occur with bad conditionning
            # or when a line of JTJ is nul
            print "LSFit encountered an error at iter = "+str(iteration)
            print "mu = "+str(mu)
            print "##### JTJ matrix #####"
            _print_info_matrix(JTJ)
            print "##### ########## #####"
            print "##### Parameter #####"
            print str(param)
            print "##### ########## #####"
            raise LinAlgError(exception_message)
        except ValueError as exception_message:
            print "LSFit encountered an error at iter = "+str(iteration)
            print "mu = "+str(mu)
            print "##### JTJ matrix #####"
            _print_info_matrix(JTJ)
            print "##### ########## #####"
            print "##### Parameter #####"
            print str(param)
            print "##### ########## #####"
            raise ValueError(exception_message)
        
        param += dparam
        
        ## Xhi square
        Xhi2 = sum(weights*(f-data)**2)
        
        ## Print Xhi square
        if debug:
            Xhi2PRINT = "[Iter="+str(iteration)+"] Xhi2 = "+str(Xhi2)
            if not LM:
                print Xhi2PRINT
            else:
                print Xhi2PRINT + " [mu="+str(mu)+"]"
                
        ## Levenberg-Marquardt update for mu
        if LM:
            f = funct(X,param)
            Xhi2_new = sum(weights*(f-data)**2)
            if Xhi2_new > Xhi2:
                mu = min(10*mu,1e10)
            else:
                mu = max(0.1*mu,1e-10)
        
        ## Stop loop based on small variation of parameter
        if sum(abs(dparam)) < dp_min*sum(abs(param)):
            stop_loop = True
            stop_trace = "Parameter not evolving enough at iter="+str(iteration)
        
        ## Stop loop based on small Jacobian
        if abs(J).sum() < J_min:
            stop_loop = True
            stop_trace = "Jacobian small enough at iter="+str(iteration)
        
        ## Increment Loop
        iteration += 1  
    
    
    ## END of LOOP and SHOW RESULTS
    
    if not quiet:
        print "[Iter="+str(iteration)+"] Xhi2 = "+str(Xhi2)
        print " "
        print "Stopping condition: "+stop_trace
        print " "
    
    return param


###############################################################################
#         Print matrix information when an error occurs                       #
###############################################################################

def _print_info_matrix(M):
    print "Matrix values:"
    for j in arange(len(M)):
        line_str = "["
        for i in arange(len(M[0])):
            line_str = line_str + " " + _num2str(M[j][i])
        print line_str + " ]"
        #print str(M[:,j])
    print "Conditioning: "+str(cond(M))
    print "Eigenvalues: "+str(eig(M)[0])

def _num2str(x):
    if x==0:
        num_str="0.00"
    elif x is inf:
        num_str = "inf"
    elif x is NaN:
        num_str = "NaN"
    else:
        power = floor(log10(abs(x)))
        str_power = str(int(power))
        if len(str_power)==1:
            str_power = "0"+str_power
        num_str = ('%.2f' % (x/10**power)) + "e" + str_power

    return num_str