#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 20:17:46 2018

@author: ivalera
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 22:40:36 2018

@author: ivalera
"""
import numpy as np
from linP import color_weight_bi_match, opt_decision
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
        thetaz = genExpertsReal(args.tau1,args.tau2, args.pBias, V, args.seed)
        
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
        
    theta_opt = opt_decision(alpha, c, pyz, z)
    #print(np.absolute((pyz[z==0]>=theta_opt[0]).sum() -(pyz[z==1]>=theta_opt[1]).sum())/N)
    #print(np.absolute((pyz[z==0]>=c).sum() - (pyz[z==1]>=c).sum())/N)
    
    
    
    
    #plt.hist(thetaz[:,0], bins='auto')
    #plt.hist(thetaz[:,1], bins='auto')  # arguments are passed to np.histogram
    #plt.show()
    
    
    Wtrue = np.zeros((N, V))
    WT = np.zeros((N, V))
    for v in range(V):
        #ind_matrix[:, v] = pyz0 >= thetaz0[v]
        indexes = pyz >= thetaz[v,z]
        Wtrue[indexes, v] = (pyz - c)[indexes]
        WT[indexes, v] = (y - c)[indexes]
    
        #W0[n,v] =  pyz0[n] - c
    
    V_nodes = ['V{}'.format(i) for i in range(V)]
    V_map = {k : v for v, k in enumerate(V_nodes)}
    
    UtilCU= np.zeros(T)
    TCU= np.zeros(T)
    Bcu0= np.zeros(T) 
    Bcu1= np.zeros(T)
    
    
    W = np.zeros((N, V))
    Z = np.zeros((N ,V))
    
    thetaMin = np.zeros((V,2));
    thetaMax = np.ones((V,2));
    thetaEst=np.zeros((V,2));
    thetaEst[range(V),0]=np.random.beta(10, 10, size=V)
    thetaEst[range(V),1]=np.random.beta(10, 10, size=V)
    #thetaEst = thetaz
    
    bb0= np.zeros(T) 
    bb1= np.zeros(T)
    
    for t in range(T):   
        for j, v in enumerate(V_nodes):
            aux=[]
            for n in range(t*M,(t+1)*M):
                d = pyz[n] >= thetaEst[j,z[n]]
                W[n, j] = (pyz[n] - c)*int(d)
                Z[n,j]= z[n]
    #           


#        bb0[t] = sum((pyz[t*M:(t+1)*M]>=theta_opt[0])*(z[t*M:(t+1)*M]==0))
#        bb1[t] = sum((pyz[t*M:(t+1)*M]>=theta_opt[1])*(z[t*M:(t+1)*M]==1))
        
        bb0[t] = sum((pyz[t*M:(t+1)*M]<theta_opt[0])*(z[t*M:(t+1)*M]==0))
        bb1[t] = sum((pyz[t*M:(t+1)*M]<theta_opt[1])*(z[t*M:(t+1)*M]==1))
        
        matchingC = color_weight_bi_match(W[t*M:(t+1)*M,:],Z[t*M:(t+1)*M,:], bb0[t], bb1[t], alpha)
        if matchingC['success']:
            matchingC= np.reshape(matchingC['x'],(M,V))
            aux1 = Wtrue[t*M:(t+1)*M,:]*(matchingC>0.5) 
            #print(aux1)
            UtilCU[t]= aux1.sum() ## Compute utilities
            aux1 = WT[t*M:(t+1)*M,:]*(matchingC>0.5) 
            #print(aux1)
            TCU[t]= aux1.sum() ## Compute utilities
            
            Bcu0[t]=((1-Z[t*M:(t+1)*M,:])*(Wtrue[t*M:(t+1)*M,:]==0)*(matchingC>0.5)).sum()
            Bcu1[t]=(Z[t*M:(t+1)*M,:]*(Wtrue[t*M:(t+1)*M,:]==0)*(matchingC>0.5)).sum()
            
            idx=np.where(matchingC==1)
            #print(idx)
            idxN = idx[0]
            idxV = idx[1]
     
            for j, (i_n, i_v) in enumerate(zip(idxN, idxV)):
                if pyz[i_n] >= thetaz[i_v,z[i_n]]:
                    thetaMax[i_v,z[i_n]] = min(1,pyz[i_n])
                else:
                    thetaMin[i_v,z[i_n]] = max(0,pyz[i_n])
                #print(i_n, i_v)
                thetaEst[i_v,z[i_n]]= (thetaMax[i_v,z[i_n]]-thetaMin[i_v,z[i_n]]) *np.random.beta(1, 1, 1) + thetaMin[i_v,z[i_n]]
        else:
            UtilCU[t]= Decimal('nan')
            TCU[t]= Decimal('nan')
            Bcu0[t]= Decimal('nan')
            Bcu1[t]= Decimal('nan') 
        
        
        
#    Dfair = np.absolute((pyz[z==0]>=theta_opt[0]).sum() -(pyz[z==1]>=theta_opt[1]).sum())/N
#    Dopt = np.absolute((pyz[z==0]>=c).sum() - (pyz[z==1]>=c).sum())/N
#    Ufair = ((pyz>=theta_opt[z])*(pyz-c)).sum()/N
#    Uopt = ((pyz>=c)*(pyz-c)).sum()/N

    result_dict = {'N':N, 'M':M, 'V':V, 'T':T, 'alpha':alpha, 'UtilCU':UtilCU, 'TCU':TCU, 'BCU1':Bcu1, 'BCU0':Bcu0}
    result_path = os.path.join(out_path, 'fair_theta_M_{}_V_{}_T_{}_alpha_{}_pBias_{}_tau_{}_it_{}.results.pkl'.format(args.M, args.V, 
                             args.T, int(args.alpha*1000), int(args.pBias*100), args.tau1, args.seed))
    with open(result_path, 'wb') as f:
        pickle.dump(result_dict, f)
    result_path = os.path.join(out_path, 'fair_theta_M_{}_V_{}_T_{}_alpha_{}_pBias_{}_tau_{}_it_{}.results.mat'.format(args.M, args.V, 
                             args.T, int(args.alpha*1000), int(args.pBias*100), args.tau1,args.seed))
    savemat(result_path, result_dict)   
    

