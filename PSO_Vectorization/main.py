from PSOKnapsak import solve_pso_knapsack
from dynamic import knapSack, knapSack_numpy
import random
import math
import numpy as np

 

def test_Knapsack():
    val = []
    wt = []
    W = 0
    t_dynamic = []
    t_pso = []
    t_dnumpy = []
    sz = []
    for i in range(1,3):
        k = 10**i
        for j in range(k,k*10,k*10):
            W = int(math.pow(k,2))
            sz.append(j)
            
            val = np.random.randint( j, dtype=int, size=j )
            wt  = np.random.randint( j%W, dtype=int, size=j )
 
            t_dynamic.append(knapSack(W, wt, val, j))
            t_dnumpy.append(knapSack_numpy(W,wt,val,j))
            t_pso.append(solve_pso_knapsack(W,wt,val,len(val),k,k if k <= 500 else 500))
            

    res = {'size':sz,'time_dynamic':t_dynamic, 'time_dynamic_numpy':t_dnumpy,'time_PSO':t_pso}
    return res
                   
print(test_Knapsack())

results_in_time = {'size': [10, 100, 1000], 'time_dynamic': [(0.0012364001013338566, 44), (0.8525676999706775, 4778), (909.0861863000318, 520493)], 'time_PSO': [(0.008936900179833174, 44), (0.9577541998587549, 4778), (89.44735640008003, 518800)]}