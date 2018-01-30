# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 18:28:23 2018

@author: rfetick
"""

from numpy import arange, meshgrid, sqrt, zeros, where, arctan, pi, cos, sin
from math import factorial

class Zernike(object):
    """
    Zernike class
    """
    
    @staticmethod
    def _checkJMN(J=None,M=None,N=None):
        if J==None and M==None and N==None:
            raise ValueError("You should define J or (M,N)")
        if J!=None:
            if M!=None or N!=None:
                raise ValueError("You cannot enter J at the same time as M or N")
        if M!=None:
            if N==None:
                raise ValueError("You should define M and N at the same time")
        if N!=None:
            if M==None:
                raise ValueError("You should define M and N at the same time")

    
    
    
    def __init__(self,Npix,J=None,M=None,N=None):

        self.Npix = Npix
        
        Zernike._checkJMN(J=J,M=M,N=N)
        
        if J!=None:
            if isinstance(J, (int, long, float)):
                self._J = [J]
            else:
                self._J = J
            self._M = []
            self._N = []
            for jj in self._J:
                MN = Zernike.JtoMN(jj)
                self._M.append(MN[0])
                self._N.append(MN[1])

        if M!=None:
            if isinstance(M, (int, long, float)):
                self._M = [M]
            else:
                self._M = M
        if N!=None:
            if isinstance(N, (int, long, float)):
                self._N = [N]
            else:
                self._N = N
                
            self._J = []
            for ii in arange(len(self._N)):
                self._J.append(Zernike.MNtoJ(self._M[ii],self._N[ii]))
                
                
        self._nbZer = len(self._J)
        self.coeffs = zeros(self._nbZer)+1.
        
        self.modes = zeros((Npix,Npix,self._nbZer))
        for ii in arange(self._nbZer):
            self.modes[:,:,ii] = Zernike.field(Npix,J=self._J[ii])
    
      
    def __repr__(self):
        return "Zernike with "+str(self._nbZer)+" modes\nJ="+self._J.__repr__()
    
    def getUnitMode(self,number):
        return self.modes[:,:,number]
         
    def getMode(self,number):
        return self.modes[:,:,number]*self.coeffs[number]
    
    def getSum(self):
        tot = zeros((self.Npix,self.Npix))
        for ii in arange(self._nbZer):
            tot += self.getMode(ii)
        return tot
    
    @staticmethod
    def JtoMN(J):
        if J < 0:
            raise ValueError("J should be positive or null")
        N=0
        while J >= N*(N+1)/2:
            N += 1
        N -= 1
        M = J - N*(N+1)/2
        M = 2*M - N
        return [M,N]
    
    @staticmethod
    def MNtoJ(M,N):
        if N < 0:
            raise ValueError("N should be positive or null")
        if abs(M)>abs(N):
            raise ValueError("abs(M) should be lower or equal to N")
        if (N-M)%2 != 0:
            raise ValueError("N minus M should be an even number")
        return N*(N+1)/2+(M+N)/2
    
    @staticmethod
    def field(Npix,M=None,N=None,J=None,circ=1):
        """
        Main method to compute a Zernike
        """
        Zernike._checkJMN(J=J,M=M,N=N)
        
        if J!=None:
            MN = Zernike.JtoMN(J)
            M = MN[0]
            N = MN[1]
        
        # COMPUTE R and THETA
        y,x = meshgrid(arange(Npix)-Npix/2,arange(Npix)-Npix/2)
        r = sqrt((x**2) + (y**2))*2./Npix
        theta = zeros((Npix,Npix))
        indices = where(x != 0)
        theta[indices] = arctan((y[indices]+0.)/(x[indices]+0.))
        
        quadrant1 = where((x <= 0) * (y > 0))
        quadrant2 = where((x < 0) * (y <= 0))
        quadrant3 = where((x >= 0) * (y < 0))
        xNull = where((x == 0) * (y >= 0))
        xyNull = where((x == 0) * (y <= 0))
    
        theta[quadrant1] = theta[quadrant1]+pi
        theta[quadrant2] = theta[quadrant2]+pi
        theta[quadrant3] = theta[quadrant3]+2*pi
        theta[xNull] = pi/2
        theta[xyNull] = 3*pi/2
        
        theta = (2*pi-theta + pi/2)%(2*pi)
        
        # COMPUTE ZERNIKE
        rmn = zeros((Npix, Npix))

        for k in arange((N-abs(M))/2 + 1):
            a = ((-1.)**k)*factorial(N-k)+0.
            b = factorial(k)*factorial((N+abs(M))/2-k)*factorial((N-abs(M))/2-k)+0.
            rmn = rmn + a/b*r**(N-2*k)
            
        if M >= 0:
            z = rmn*cos(M*theta)
        else:
            z = rmn*sin(abs(M)*theta)
            
        if circ == 1:
            z *= (r <= 1)
            z = z/sqrt((z**2).sum())
        
        return z
    
    @staticmethod
    def JtoSTR(J):
        if J==0:
            return "piston"
        elif J==1:
            return "tip-tilt X"
        elif J==2:
            return "tip-tilt Y"
        elif J==3:
            return "astigmatism"
        elif J==4:
            return "defocus"
        elif J==5:
            return "astigmatism"
        elif J==6:
            return "trefoil"
        elif J==7:
            return "coma"
        elif J==8:
            return "coma"
        elif J==9:
            return "trefoil"
        
        
        