#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 11:42:47 2018

@author: ivalera
"""
import numpy as np
import scipy.optimize 

def weight_bi_match(W):
    N,V=W.shape;
    #Wvec = matrix(np.array(-W).flatten())
    Wvec = (np.array(-W).flatten())
    #Ce= np.zeros((V+N,N*V))
    #Cg=(-np.eye(N*V))
    Ce= np.zeros((N,N*V))
    Cg=np.zeros((N*V+V, N*V))
    Cg[0:N*V, 0:N*V]=(-np.eye(N*V))
    for n in range(N):
        Ce[n, n*V:(n+1)*V]= np.ones(V) # sum_v x_uv =1 for all u
        #Ce[N:N+V, n*V:(n+1)*V]= np.eye(V) #sum_u x_uv <=1
        Cg[N*V:N*V+V, n*V:(n+1)*V]= np.eye(V) #sum_u x_uv <=1
        
#    be=(np.ones(V+N))
#    bg= (np.zeros(N*V))
    be=(np.ones(N))
    bg= np.concatenate((np.zeros(N*V), np.ones(V) ),axis=0)

    matching=scipy.optimize.linprog(Wvec,Cg,bg,Ce,be) 
    return matching

def color_weight_bi_match(W, Z, b0, b1, alpha):
    N,V=W.shape;
    #Wvec = matrix(np.array(-W).flatten())
    Wvec = (np.array(-W).flatten())
    Ce= np.zeros((N,N*V))
    Cg=np.zeros((N*V+V, N*V))
    Cg[0:N*V, 0:N*V]=(-np.eye(N*V))
    for n in range(N):
        Ce[n, n*V:(n+1)*V]= np.ones(V) # sum_v x_uv =1 for all u
        #Ce[N:N+V, n*V:(n+1)*V]= np.eye(V) #sum_u x_uv <=1
        Cg[N*V:N*V+V, n*V:(n+1)*V]= np.eye(V) #sum_u x_uv <=1
        
    W0= (W==0)
    W1= (W!=0)
#    C00= (1-Z)*W0
#    C01 = (1-Z)*W1
#    C10 = Z*W0;
#    C11 = Z*W1;
    
    C00= (1-Z)*W1
    C01 = (1-Z)*W0
    C10 = Z*W1;
    C11 = Z*W0;
            
    Cg= np.concatenate((Cg, np.reshape(C00,(1,N*V))),axis=0)
    Cg= np.concatenate((Cg, np.reshape(C01,(1,N*V))),axis=0)
    Cg= np.concatenate((Cg, np.reshape(C10,(1,N*V))),axis=0)
    Cg= np.concatenate((Cg, np.reshape(C11,(1,N*V))),axis=0)

    be=(np.ones(N))
    bg= np.concatenate((np.zeros(N*V), np.ones(V) ),axis=0)
    bg= np.concatenate((bg, np.array([-b0+(1+alpha)*sum(1-Z[:,1]), b0+alpha*sum(1-Z[:,1]), -b1+(1+alpha)*sum(Z[:,1]), b1+alpha*sum(Z[:,1])])),axis=0)

    matching=scipy.optimize.linprog(Wvec,Cg,bg,Ce,be) 
    return matching #np.reshape(matching['x'],(N,V))


    
def opt_decision(alpha, c, pyz, z):
    nV=100
    N=z.shape[0]
    util0=np.zeros(nV)
    util1=np.zeros(nV)
    b0=np.zeros(nV)
    b1=np.zeros(nV)
    Vtheta= np.array(range(nV))/nV
    for tt in range(nV):
        util0[tt]= ((pyz[z==0]>=Vtheta[tt])*(pyz[z==0]-c)).sum()
        util1[tt]= ((pyz[z==1]>=Vtheta[tt])*(pyz[z==1]-c)).sum()
#        b0[tt]= ((pyz[z==0]>=Vtheta[tt])).sum()
#        b1[tt]= ((pyz[z==1]>=Vtheta[tt])).sum()
        b0[tt]= ((pyz[z==0]<Vtheta[tt])).sum()
        b1[tt]= ((pyz[z==1]<Vtheta[tt])).sum()

    Disp = np.absolute(np.tile(b0,(nV,1)).T-(np.tile(b1,(nV,1))))/N
    Util = (np.tile(util0,(nV,1))+(np.tile(util1,(nV,1)).T))/N
    idx = np.argmax((Disp<=alpha)*Util)
    [idx0, idx1] = np.unravel_index(idx, (nV,nV))

    Theta= np.array([Vtheta[idx0], Vtheta[idx1]])
#    print(np.absolute((pyz[z==0]>=Vtheta[idx0]).sum() - (pyz[z==1]>=Vtheta[idx1]).sum())/N, Disp[idx0,idx1])
#    print(np.absolute((pyz[z==0]>=c).sum() - (pyz[z==1]>=c).sum())/N)         
    return Theta


#def color_weight_bi_match(W, Z, b0, b1, alpha):
#    N,V=W.shape;
#    #Wvec = matrix(np.array(-W).flatten())
#    Wvec = (np.array(-W).flatten())
#    Ce= np.zeros((V+N,N*V))
#    for n in range(N):
#        Ce[n, n*V:(n+1)*V]= np.ones(V)
#        Ce[N:N+V, n*V:(n+1)*V]= np.eye(V)
#    W0= (W==0)
#    W1= (W!=0)
#    C00= (1-Z)*W0
#    C01 = (1-Z)*W1
#    C10 = Z*W0;
#    C11 = Z*W1;
#        
#    Cg=(-np.eye(N*V))
#    Cg= np.concatenate((Cg, np.reshape(C00,(1,N*V))),axis=0)
#    Cg= np.concatenate((Cg, np.reshape(C01,(1,N*V))),axis=0)
#    Cg= np.concatenate((Cg, np.reshape(C10,(1,N*V))),axis=0)
#    Cg= np.concatenate((Cg, np.reshape(C11,(1,N*V))),axis=0)
#
#    be=(np.ones(V+N))
#    bg= np.concatenate((np.zeros(N*V), np.array([-b0+(1+alpha)*sum(1-Z[:,1]), b0+alpha*sum(1-Z[:,1]), -b1+(1+alpha)*sum(Z[:,1]), b1+alpha*sum(Z[:,1])])),axis=0)
#
#    matching=scipy.optimize.linprog(Wvec,Cg,bg,Ce,be) 
#    return np.reshape(matching['x'],(N,V))
