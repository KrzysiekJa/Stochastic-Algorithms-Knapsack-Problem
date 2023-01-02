import random 
import matplotlib as mp 
import numpy as np
import sys
from decorator import bench


W = 5
wt = [1,2,10,200,3]
val = [2,5,6,1,100]

# W = maximum capacity
# wt = weight
# val = value
# n = size of the objects
# s = size of particles

#introduce a flexible penality, exponential scaling, take the exponetial value instead the normal fitness it should grow faster
def fitness_function( x, w, v, W ):
    if( np.dot(x,w) <= W ):
        return np.dot(x,v)
    else:
        return 0


def solve_pso_knapsack( W, wt, vals, n, n_particles ):
    epochs = 5
    swarm = []
    global_best_val = -1
    global_best = np.zeros(n, dtype=np.int8)
    sigmoid = lambda x :  1/(1 + np.exp(-x))

    X=0
    PLBEST =1
    VELOCITY_Y =2

    
    C1 = 2.05
    C2 = 2.05

    
    # Random Generating the xi in the particles
    # Initializing the local minimum in the swarm
    # find the global solution inside the swarm
    for _ in range(n_particles):
        # particles has x , p, velocity
        pi = ([],[],[])

        position_i = np.random.randint( 2, dtype=np.int8, size=n ) # random binary array
        ## temporary_i = random_binary_list(n)
        tmp_val = fitness_function( position_i, wt, vals, W )
        
        if( tmp_val > global_best_val ):
            global_best_val = tmp_val
            np.copyto( global_best, position_i ) # syntax: np.copyto(dst, src)
        
        velocity_i = np.zeros(n)
        swarm.append( (position_i, position_i, velocity_i) )
    
    print( ">> ", global_best_val )
    print( "## ", global_best )
    
    for _ in range(epochs):
        for i in range(n_particles):
            
            # checking PLBEST
            if fitness_function( swarm[i][X], wt, vals, W ) > fitness_function( swarm[i][PLBEST], wt, vals, W ):
                np.copyto( swarm[i][PLBEST], swarm[i][X] )
                # for d in range(n):
                #     swarm[i][PLBEST][d] = swarm[i][X][d]
            
            # checking global_best
            if fitness_function( swarm[i][PLBEST], wt, vals, W ) > global_best_val:
                np.copyto( global_best, swarm[i][PLBEST] )
                # for d in range(n):
                #     global_best[d] = swarm[i][PLBEST][d]
                global_best_val = fitness_function( global_best, wt, vals, W ) # update
            
            ro = np.random.randint( 2, dtype=np.int8, size=n ) # random binary array

            for d in range(n):
                swarm[i][VELOCITY_Y][d] += C1 * random.random() * (swarm[i][PLBEST][d]-swarm[i][X][d])  + C2 * random.random() * (global_best[d]-swarm[i][X][d])
                
                if ro[d] < sigmoid( swarm[i][VELOCITY_Y][d] ):
                    swarm[i][X][d]=1
                else:
                    swarm[i][X][d]=0
            
            # genetic mutations
            nmut = random.randrange(n) + 1
            for j in range(nmut):
                k = random.randrange(n)
                swarm[i][X][k] = swarm[i][PLBEST][k]
            
            nmut = random.randrange(n) + 1
            for j in range(nmut):
                k = random.randrange(n)
                swarm[i][X][k] = global_best[k]
            
            print( ">> ", global_best_val )
            print( "## ", global_best )


solve_pso_knapsack( 10, [1,20,10,2,2], [5,100,10,100,1], 5, 2 )


@bench
def test_Knapsack():
    val = []
    wt = []
    W = 50
    ti = []
    sz = []
    for i in range(1,6):
        k = 10**i
        for j in range(k,k*10,k*10):
            val = []
            wt = []
            sz.append(j)
            for t in range(0,j):
                val.append(random.randint(0,j))
                wt.append(random.randint(0,j)%W)
            ti.append(solve_pso_knapsack(W, wt, val, j, int( np.sqrt(k)) ))
    res = {'size':sz,'time':ti}
    return res
                   
#print(test_Knapsack())

