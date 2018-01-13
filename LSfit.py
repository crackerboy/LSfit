# -*- coding: utf-8 -*-
"""
Created on Tue Jan 02 17:57:41 2018
Updated on Sat Jan 13 13:01:00 2018
    Removed these ugly eval(.)
    Created class LSparam for bounding and fixing parameters

INFORMATIONS
Errors might occur due to bad matrix conditioning.
This can be due to low number of data
Please be careful with that.

TO-DO
- Create a LSfit2D(.)
- Check Levenberg-Marquardt regularization
- Check the WEIGHTED least-square convergence
- Errors mights occur sometimes due to bad matrix conditioning
    Improve matrix conditioning when (or before) a LinAlgError is raised
    
Licence GNU-GPL v3.0

@author: rfetick
"""

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
     param0  - [Vector,LSparam] First estimation of parameters
                 param0 can be simply a vector
                 if param0 is a LSparam, you can define bounds, fixed values...
    
    ##### OPTIONAL INPUTS #####
     weights - [Vector] Weights (1/noise^2) on each coordinate X
                  Default : no weighting (weights=1)
                  weights = 1/sigma^2 for a gaussian noise
                  weights = 1/data for a Poisson-like noise
     LM      - [Boolean] Levenberg-Marquardt algorithm
                  Default : LM = False
     quiet   - [Boolean] Don't print status of algorithm
                  Default : quiet = False
     debug   - [Integer] Print status for each iteration
                  Default : debug = 0
                  For example debug = 5 prints 1 line over 5
    
    #####  OUTPUT  #####
     param   - [Vector] Best parameters for least-square fitting
    """
    # INITIALIZATION
    sX = size(X)
    param = LSparam(param0)

    J = zeros((sX,param.nb_param)) # Jacobian
    h = 1e-9 #step to compute Jacobian
    
    
    # STOPPING CONDITIONS
    J_min = 1e-5*param.nb_param**2 # stopping criteria based on small Jacobian
    dp_min = 1e-8 # stopping criteria based on small variation of parameters
    max_iter = 1e4 # stopping criteria based on max number of iteration
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
        print param.value
        f = funct(array(X),param.value)
        Xhi2 = sum(weights*(f-data)**2)
        print "[Iter=0] Xhi2 = "+str(Xhi2)
    
    
    # LOOP
    
    iteration = 0
    
    while (iteration < max_iter) and not stop_loop:
        
        f = funct(array(X),param.value)
        
        ## Iterate over each parameter to compute derivative
        for ip in arange(param.nb_param):
            
            J[:,ip] = weights*(funct(array(X),param.value+h*param.one(ip))-f)/h
        
        ## Compute dvalue = -{(transpose(J)*J)^(-1)}*transpose(J)*(func-data)
        JTJ = dot(J.T,J)
        try:
            param.dvalue = - dot(dot(inv(JTJ+mu*diag(JTJ.diagonal())),J.T),weights*(f-data))
        except LinAlgError as exception_message:
            # These errors occur with bad conditionning
            # or when a line of JTJ is nul
            print "LSFit encountered an error at iter = "+str(iteration)
            print "mu = "+str(mu)
            print "##### JTJ matrix #####"
            _print_info_matrix(JTJ)
            print "##### ########## #####"
            print "##### Parameter #####"
            print str(param.value)
            print "##### ########## #####"
            raise LinAlgError(exception_message)
        except ValueError as exception_message:
            print "LSFit encountered an error at iter = "+str(iteration)
            print "mu = "+str(mu)
            print "##### JTJ matrix #####"
            _print_info_matrix(JTJ)
            print "##### ########## #####"
            print "##### Parameter #####"
            print str(param.value)
            print "##### ########## #####"
            raise ValueError(exception_message)
        
        # Step forward with dvalue
        param.step()
        
        ## Xhi square
        Xhi2 = sum(weights*(f-data)**2)
        
        ## Print Xhi square
        if debug and (iteration % debug)==0:
            print "[Iter="+str(iteration)+"] Xhi2 = "+str(Xhi2)
            if LM:
                print "[Iter="+str(iteration)+"] mu = "+str(mu)
            print "[Iter="+str(iteration)+"] Conditioning = "+_num2str(cond(JTJ))
                
        ## Levenberg-Marquardt update for mu
        if LM:
            f = funct(X,param.value)
            Xhi2_new = sum(weights*(f-data)**2)
            if Xhi2_new > Xhi2:
                mu = min(10*mu,1e10)
            else:
                mu = max(0.1*mu,1e-10)
        
        ## Stop loop based on small variation of parameter
        if sum(abs(param.dvalue)) < dp_min*sum(abs(param.value)):
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
    
    return param.value


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

    print "Conditioning: "+_num2str(cond(M))
    
    line_str = "["
    for i in arange(len(M)):
        line_str = line_str + " " + _num2str(eig(M)[0][i])
    print "Eigenvalues: " + line_str + " ]"

def _num2str(x):
    if x==0:
        num_str="0.00"
    elif x is inf or x == inf:
        num_str = "inf"
    elif x is NaN:
        num_str = "NaN"
    else:
        power = floor(log10(abs(x)))
        str_power = str(int(power))
        if len(str_power)==1:
            str_power = "0" + str_power
        if power>=0:
            str_power = "+" + str_power
        if power<0 and len(str_power)==2:
            str_power = "-0"+str_power[1]
        num_str = ('%.2f' % (x/10**power)) + "e" + str_power

    return num_str

###############################################################################
#                   Define a class to precise parameters                      #
###############################################################################
    
class LSparam(object):
    """
    Class LS param, to be used is LSfit
    Define more precisely your parameters
    
    ATTRIBUTES
    value         - [List] Initial guess for your parameters
    fixed         - [List] Set to 'True' if parameters are fixed
    bound_up      - [List] Set the value of eventual up-bounds
    bound_down    - [List] Set the value of eventual down-bounds
    is_bound_up   - [List] Set to 'True' to activate bounds
    is_bound_down - [List] Set to 'True' to activate bounds
    
    """
    
    def __init__(self,value):
        if isinstance(value,LSparam):
            self.copyLSparam(value)
        else:
            if isinstance(value, (int, long, float, complex)):
                valueList = [value]
            else:
                valueList = value
            L = len(valueList)
            # User can influence these attributes
            self.value = array(valueList,dtype=float)
            self.fixed = [False for i in arange(L)]
            self.bound_up = array([0 for i in arange(L)],dtype=float)
            self.bound_down = array([0 for i in arange(L)],dtype=float)
            self.is_bound_up = [False for i in arange(L)]
            self.is_bound_down = [False for i in arange(L)]
            # User shouldn't access these attributes
            self.nb_param = L
            self.valueInit = self.value
            self.valueOld = self.value
            self.dvalue = array([0 for i in arange(L)],dtype=float)
        
    
    def copyLSparam(self,objToCopy):
        """
        Copy old LSparam into current one
        """
        all_attr = objToCopy.__dict__
        for key in all_attr:
            setattr(self,key,getattr(objToCopy,key))
    
    def show(self):
        """
        Display information
        """
        print "########## LSparam ##########"
        print "Values         : " + str(self.value)
        print "Fixed          : " + str(self.fixed)       
        print "Bounds up      : " + str(self.bound_up)
        print "Is bounded up  : " + str(self.is_bound_up)       
        print "Bounds down    : " + str(self.bound_down)
        print "Is bounded down: " + str(self.is_bound_down)
        print "#############################"
        
    def one(self,i):
        """
        Returns a vector of size nb_param
        This vector is nul, excepted the i-th component equals 1
        Allows to compute partial derivatives for example
        """
        a = zeros(self.nb_param)
        a[i] = 1.0
        return a
    
    def step(self):
        """
        Steps forward the param.value with param.dvalue
        """
        self.valueOld = self.value
        # Before stepping, check if we are inside the bounds
        conditionUP = ((self.value + self.dvalue) < self.bound_up) | (1- array(self.is_bound_up))
        conditionDOWN = ((self.value + self.dvalue) > self.bound_down) | (1- array(self.is_bound_down))
        conditionFIXED = 1 - array(self.fixed)
        self.value = self.value + self.dvalue * (conditionUP & conditionDOWN & conditionFIXED)
        
    def check(self):
        """
        Check values consistency
        """
        if inf in self.value:
            return False
        if NaN in self.value:
            return False
        return True
        
    def set_bound_down(self,val):
        """
        Set all down bounds with same value
        """
        self.bound_down = val*(zeros(self.nb_param)+1)
        self.is_bound_down = [True for i in arange(self.nb_param)]
        
    def set_bound_up(self,val):
        """
        Set all up bounds with same value
        """
        self.bound_up = val*(zeros(self.nb_param)+1)
        self.is_bound_up = [True for i in arange(self.nb_param)]
