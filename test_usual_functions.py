# -*- coding: utf-8 -*-
"""
Created on Wed Jan 10 23:10:48 2018

Test usual functions

@author: rfetick
"""

## IMPORT

from numpy import *
from pylab import *

## Test Gauss 1D and Moffat 1D
X = linspace(0,10,num=100)
bck = 1.
amp = 1.
center = 5.
G = gauss(X,[bck,amp,1.,center])
M = moffat(X,[bck,amp,1.,center,2.])
plot(X,G,linewidth=2)
plot(X,M,linewidth=2)
legend(("Gauss","Moffat"))
title("1D Gauss and 1D Moffat")
show()

## Test Gauss 2D
xmax = 100
[X,Y] = mgrid[0:xmax,0:xmax]
A = zeros(7)
A[1] = 1
A[2] = 10
A[3] = 10
A[4] = xmax/2
A[5] = xmax/2
g = gauss2D(X,Y,A)
pcolor(g)
axis('equal')
title("2D Gauss")
show()

## Test Moffat 2D
xmax = 100
[X,Y] = mgrid[0:xmax,0:xmax]
A = zeros(8)
A[1] = 1
A[2] = 10
A[3] = 10
A[4] = xmax/2
A[5] = xmax/2
A[7] = 2
m = moffat2D(X,Y,A)
pcolor(m)
axis('equal')
title("2D Moffat")
show()