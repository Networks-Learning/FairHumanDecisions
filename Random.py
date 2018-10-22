#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 19:41:09 2018

@author: ivalera
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 22:40:36 2018

@author: ivalera
"""
import numpy as np
from linP import weight_bi_match
from genData import genCases, genExperts, genRealData, genExpertsReal
from decimal import Decimal
import os
import argparse
import pickle
from scipy.io import savemat


if __name__ == '__main__':

    #########################################
    # creating the opt parser
    parser = argparse.ArgumentParser()

    
    parser.add_argument('-o', '--output', type=str,
                        default='./exp/',
                        help='Output path to dataset generation result')
    
    parser.add_argument('--data', '--real', type=bool,
                        default= False,
                        help='Real or synthetic data')
    
    parser.add_argument('--pTest',  type=float,
                        default=0.5,
                        help='% of test data')
    
    parser.add_argument('--pBias',  type=float,
                        default=0.5,
                        help='% of test data')
    
    parser.add_argument('--tau1',  type=float,
                        default=1,
                        help='% param beta')
    
    parser.add_argument('--tau2',  type=float,
                        default=1,
                        help='% param beta')
    

    parser.add_argument('-V', type=int,
                        default=3,
                        help='Number of judges')
    
    parser.add_argument('-T',  type=int,
                        default=1000,
                        help='Number of rounds')
    
        
    parser.add_argument('-M',  type=int,
                        default=20,
                        help='Number of rounds')
    
    parser.add_argument('--alpha',  type=float,
                        default=1,
                        help='Number of rounds')

    parser.add_argument('--seed', type=int, nargs='?',
                        default=0,
                        help='Seed for the random generator')

    #
    # parsing the args
    args = parser.parse_args()
    np.random.seed(args.seed)

    c=0.5
    M = args.M #number of rounds
    V = (args.V)*M # number of Judges 
    alpha = args.alpha # number of Judges 
    if args.data:
                  
        out_path = os.path.join('./Real/', str(args.seed))
        os.makedirs(out_path, exist_ok=True)
        z, y, pyz = genRealData(args.pTest,args.seed)
        N = len(z)
        T = int(np.floor(N/M))
        N=T*M;
        z=z[0:N]
        y=y[0:N]
        pyz=pyz[0:N]
        
        # Generate experts
        thetaz =genExpertsReal(args.tau1,args.tau2,args.pBias, V, args.seed)
        
    else:
            
        out_path = os.path.join(args.output, str(args.seed))
        os.makedirs(out_path, exist_ok=True)
        T = args.T #number of rounds
        N= M*T # number of cases
  
        #alpha=0.01
        
        #Same distribution for males and females
        z, y, pyz =genCases(N, args.seed)  
    
        # Generate experts
        thetaz =genExperts(V, args.seed)
    #plt.hist(thetaz[:,0], bins='auto')
    #plt.hist(thetaz[:,1], bins='auto')  # arguments are passed to np.histogram
    #plt.show()
    
    
    Wtrue = np.zeros((N, V))
    WT = np.zeros((N, V))
    
    for v in range(V):
        #ind_matrix[:, v] = pyz0 >= thetaz0[v]
        indexes = pyz >= thetaz[v,z]
        Wtrue[indexes, v] = (pyz - c)[indexes]
        WT[indexes, v]= (y - c)[indexes]
    
        #W0[n,v] =  pyz0[n] - c
    
    V_nodes = ['V{}'.format(i) for i in range(V)]
    V_map = {k : v for v, k in enumerate(V_nodes)}
    
    nR=10;
    UtilR= np.zeros((nR,T))
    TR= np.zeros((nR,T))
    Br0= np.zeros((nR,T))
    Br1= np.zeros((nR,T))
    
    W = np.zeros((N, V))
    Z = np.zeros((N ,V))
    
    thetaMin = np.zeros((V,2));
    thetaMax = np.ones((V,2));
    thetaEst=np.zeros((V,2));
    thetaEst[range(V),0]=np.random.beta(10, 10, size=V)
    thetaEst[range(V),1]=np.random.beta(10, 10, size=V)
    #thetaEst = thetaz
    
    for t in range(T):   
        for j, v in enumerate(V_nodes):
            aux=[]
            for n in range(t*M,(t+1)*M):
#            d = pyz[n] >= thetaEst[j,z[n]]
#            W[n, j] = (pyz[n] - c)*int(d)
                Z[n,j]= z[n]
        for nn in range(nR):        
            matchingR=  np.zeros((M,V));
            idxN = np.random.permutation(M);
            idxV = np.random.permutation(V);
            matchingR[idxN,idxV[0:M]] = 1 
            aux = Wtrue[t*M:(t+1)*M,:]*matchingR
            UtilR[nn,t]= aux.sum() ## Compute utilities
            aux = WT[t*M:(t+1)*M,:]*matchingR
            TR[nn,t]= aux.sum() ## Compute utilities
            
            Br0[nn,t]=((1-Z[t*M:(t+1)*M,:])*(Wtrue[t*M:(t+1)*M,:]==0)*(matchingR>0.5)).sum()
            Br1[nn,t]=(Z[t*M:(t+1)*M,:]*(Wtrue[t*M:(t+1)*M,:]==0)*(matchingR>0.5)).sum()
                

    result_dict = {'N':N, 'M':M, 'V':V, 'T':T, 'alpha':alpha, 'UtilR':UtilR,  'TR':TR, 
                       'BR0':Br0, 'BR1':Br1}
    result_path = os.path.join(out_path, 'random_M_{}_V_{}_T_{}_pBias_{}_tau_{}_it_{}.results.pkl'.format(args.M, args.V, 
                             args.T, int(args.pBias*100),  args.tau1, args.seed))
    with open(result_path, 'wb') as f:
        pickle.dump(result_dict, f)
    result_path = os.path.join(out_path, 'random_M_{}_V_{}_T_{}_pBias_{}_tau_{}_it_{}.results.mat'.format(args.M, args.V, 
                             args.T, int(args.pBias*100), args.tau1, args.seed))
    savemat(result_path, result_dict)
   