# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 23:17:57 2018

@author: rfetick
"""

from numpy import exp, cos, sin

###############################################################################
###############################################################################

def gauss(X,A):
    """
    ### USAGE ###
    Create a 1D Gaussian function
    ### INPUTS ###
    X are the coordinates where to evaluate the function
    A is the voctor of parameter defined as following
    A[0]   Constant baseline level
    A[1]   Peak value
    A[2]   Sigma
    A[3]   Peak centroid
    """
    return A[1]*exp(-((X-A[3])**2)/(2*A[2]**2))+A[0]

###############################################################################
###############################################################################

def gauss2D(X,Y,A):
    """
    ### USAGE ###
    Create a 2D Gaussian function
    ### INPUTS ###
    [X,Y] is the meshgrid where to evaluate the function
    A is the voctor of parameter defined as following
    A[0]   Constant baseline level
    A[1]   Peak value
    A[2]   Sigma X
    A[3]   Sigma Y
    A[4]   Peak centroid (x)
    A[5]   Peak centroid (y)
    A[6]   Tilt angle (clockwise)
    """
    
    alpha_X = A[2]
    alpha_Y = A[3]
    
    # Rotational angles
    xNum = (cos(A[6])/alpha_X)**2 + (sin(A[6])/alpha_Y)**2
    yNum = (cos(A[6])/alpha_Y)**2 + (sin(A[6])/alpha_X)**2
    xyNum = sin(2*A[6])/alpha_Y**2 - sin(2*A[6])/alpha_X**2

    # Compute Moffat
    u  = xNum*(X-A[4])**2 + xyNum*(X-A[4])*(Y-A[5]) + yNum*(Y-A[5])**2
    
    return A[1]*exp(-.5*u)+A[0]

###############################################################################
###############################################################################

def moffat(X,A):
    """
    ### USAGE ###
    Create a 1D Moffat function
    ### INPUTS ###
    X are the coordinates where to evaluate the function
    A is the voctor of parameter defined as following
    A[0]   Constant baseline level
    A[1]   Peak value
    A[2]   Peak half-width
    A[3]   Peak centroid
    A[4]   Moffat power law
    ### OUTPUT ###
    Y the Moffat function evaluated on the X coordinates
    """

    # Compute Moffat
    u  = ((X-A[3])/A[2])**2
    moff = A[1]/(u + 1.)**A[4] + A[0]
    
    return moff

###############################################################################
###############################################################################

def moffat2D(X,Y,A):
    """
    ### USAGE ###
    Create a 2D Moffat function
    ### INPUTS ###
    [X,Y] is the meshgrid where to evaluate the function
    A is the voctor of parameter defined as following
    A[0]   Constant baseline level
    A[1]   Peak value
    A[2]   Peak half-width (ALPHA x)
    A[3]   Peak half-width (ALPHA y)
    A[4]   Peak centroid (x)
    A[5]   Peak centroid (y)
    A[6]   Tilt angle (clockwise)
    A[7]   Moffat power law
    ### OUTPUT ###
    The 2D-Moffat evaluated on the meshgrid
    """
    
    alpha_X = A[2]
    alpha_Y = A[3]
    
    # Rotational angles
    xNum = (cos(A[6])/alpha_X)**2 + (sin(A[6])/alpha_Y)**2
    yNum = (cos(A[6])/alpha_Y)**2 + (sin(A[6])/alpha_X)**2
    xyNum = sin(2*A[6])/alpha_Y**2 - sin(2*A[6])/alpha_X**2

    # Compute Moffat
    u  = xNum*(X-A[4])**2 + xyNum*(X-A[4])*(Y-A[5]) + yNum*(Y-A[5])**2
    moff = A[1]/(u + 1.)**A[7] + A[0]
    
    return moff