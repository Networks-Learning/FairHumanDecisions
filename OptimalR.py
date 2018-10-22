#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 11:24:44 2018

@author: ivalera
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 12:34:33 2018

@author: ivalera
"""
import numpy as np
from linP import  opt_decision
from genData import genCases, genRealData
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
        print(T)
        
    else:
            
        out_path = os.path.join(args.output, str(args.seed))
        os.makedirs(out_path, exist_ok=True)
        T = args.T #number of rounds
        N= M*T # number of cases
  
        #alpha=0.01
        
        #Same distribution for males and females
        z, y, pyz =genCases(N, args.seed)  
        
    theta_opt = opt_decision(alpha, c, pyz, z)
    print(theta_opt)

    Uopt= np.zeros(T)
    Topt= np.zeros(T)
    Bopt0= np.zeros(T) 
    Bopt1= np.zeros(T) 
    
    Ufair= np.zeros(T)
    Tfair= np.zeros(T)
    Bfair0= np.zeros(T) 
    Bfair1= np.zeros(T)
    

    for t in range(T):   
        
        Ufair[t] = ((pyz[t*M:(t+1)*M]>=theta_opt[z[t*M:(t+1)*M]])*(pyz[t*M:(t+1)*M]-c)).sum()
        Uopt[t] = ((pyz[t*M:(t+1)*M]>=c)*(pyz[t*M:(t+1)*M]-c)).sum()
        
        Tfair[t] = ((pyz[t*M:(t+1)*M]>=theta_opt[z[t*M:(t+1)*M]])*(y[t*M:(t+1)*M]-c)).sum()
        Topt[t] = ((pyz[t*M:(t+1)*M]>=c)*(y[t*M:(t+1)*M]-c)).sum()
        
        
#        Bopt0[t] = sum((pyz[t*M:(t+1)*M]>=c)*(z[t*M:(t+1)*M]==0))
#        Bopt1[t] = sum(1-(pyz[t*M:(t+1)*M]>=c)*(z[t*M:(t+1)*M]==1))
#        
#        Bfair0[t] = sum((pyz[t*M:(t+1)*M]<theta_opt[0])*(z[t*M:(t+1)*M]==0))
#        Bfair1[t] = sum((pyz[t*M:(t+1)*M]<theta_opt[1])*(z[t*M:(t+1)*M]==1))
        
        Bopt0[t] = sum((pyz[t*M:(t+1)*M]<c)*(z[t*M:(t+1)*M]==0))
        Bopt1[t] = sum((pyz[t*M:(t+1)*M]<c)*(z[t*M:(t+1)*M]==1))
        
        Bfair0[t] = sum((pyz[t*M:(t+1)*M]<theta_opt[0])*(z[t*M:(t+1)*M]==0))
        Bfair1[t] = sum((pyz[t*M:(t+1)*M]<theta_opt[1])*(z[t*M:(t+1)*M]==1))

    #print(np.absolute((pyz[z==0]<c).sum()-(pyz[z==1]<c).sum())/N)
   # print(np.absolute((pyz[z==0]< theta_opt[0]).sum()-(pyz[z==1]<theta_opt[1]).sum())/N)
    print(np.absolute(Bopt0.sum()-Bopt1.sum())/N, np.absolute(Bfair0.sum()-Bfair1.sum())/N)
    print(Uopt.sum()/N, Ufair.sum()/N)
    
    result_dict = {'N':N, 'M':M, 'V':V, 'T':T, 'alpha':alpha,'Bopt0':Bopt0, 'Bopt1':Bopt1, 'Uopt':Uopt,'Topt':Topt,
                           'Bfair0':Bfair0, 'Bfair1':Bfair1, 'Ufair':Ufair, 'Tfair':Tfair}
    result_path = os.path.join(out_path, 'opt_M_{}_V_{}_T_{}_alpha_{}_pBias_{}_it_{}.results.pkl'.format(args.M, args.V, 
                             args.T, int(args.alpha*1000), int(args.pBias*100), args.seed))
    with open(result_path, 'wb') as f:
        pickle.dump(result_dict, f)
    result_path = os.path.join(out_path, 'opt_M_{}_V_{}_T_{}_alpha_{}_pBias_{}_it_{}.results.mat'.format(args.M, args.V, 
                             args.T, int(args.alpha*1000),int(args.pBias*100), args.seed))
    savemat(result_path, result_dict)   
    

