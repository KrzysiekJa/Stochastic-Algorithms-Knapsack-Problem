from PSOKnapsak import solve_pso_knapsack
from dynamic import knapSack, knapSack_numpy,knapSack_numpy2
import random
import math
import numpy as np

 

def test_Knapsack():
    val = []
    wt = []
    W = 0
    #t_dynamic = []
    t_pso = []
    t_dnumpy2 = []
    sz = []
    for i in range(1,3):
        k = 10**i
        for j in range(k,k*10,k*10):
            W = int(math.pow(k,2))
            sz.append(j)
            # Genereting two random vector {+1 for no zero elements}
            val = np.random.randint( j, dtype=int, size=j )+1
            wt  = np.random.randint( j%W, dtype=int, size=j )+1
            #t_dynamic.append(knapSack(W, wt, val, j))

            t_dnumpy2.append(knapSack_numpy2(W,wt,val,j))
            optimal = t_dnumpy2[i-1][1]
            #Doing different tests on the same data on the pso algorithm
            values = []
            times = []
            #Repeated test with the same imput values
            for q in range(0,50):
                res = solve_pso_knapsack(W,wt,val,len(val),10,k*2 if k <= 500 else 500)
                values.append(res[1])
                times.append(res[0])
            #Computing the mean on values, the standard deviation and the gap with the optimal solution
            mean = np.mean(values)
            mean_t = np.mean(times)
            std = np.std(values)
            gap = (optimal-mean)/optimal
            #The final result in this case is a dictionary with the means of the results
            t_pso.append({'time_m':mean_t,'value_m':mean,'value_std':std,'gap':gap})
            print(t_dnumpy2)

    res = {'size':sz, 'time_numpy2':t_dnumpy2,'time_PSO':t_pso}
    return res
                   
print(test_Knapsack())

# results = {'size': [10, 100, 1000], 'time_numpy2': [(9.640003554522991e-05, 55), (0.0019996000919491053, 5057), (11.202072399901226, 507972)], 'time_PSO': [{'time_m': 0.01412677600979805, 'value_m': 54.94, 'value_std': 0.42000000000000004, 'gap': 0.0010909090909091322}, {'time_m': 0.15821873203385622, 'value_m': 4970.46, 'value_std': 44.0046406643663, 'gap': 0.01711291279414672}, {'time_m': 0.7504494219878688, 'value_m': 397979.18, 'value_std': 4518.877792948156, 'gap': 0.2165332341152662}]}