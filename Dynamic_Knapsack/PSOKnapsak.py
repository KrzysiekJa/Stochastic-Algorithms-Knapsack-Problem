import sys
from random import random, randrange
import numpy as np
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

def sigmoid( x ):
    return 1/(1 + np.exp(-x))


def solve_pso_knapsack( W, wt, vals, n, n_particles, epochs = 5, C1 = 2.0, C2 = 2.0):
    particle_params = ['POSITION', 'P_LOCAL_BEST', 'VELOCITY']
    GLOBAL_BEST_VAL = -1
    GLOBAL_BEST = np.zeros(n, dtype=np.int8)

    swarm = []
    # Random Generating the xi in the particles
    # Initializing the local minimum in the swarm
    # find the global solution inside the swarm
    for _ in range(n_particles):
        position = np.random.randint( 2, dtype=np.int8, size=n ) # random binary array
        tmp_val = fitness_function( position, wt, vals, W )
        
        if( tmp_val > GLOBAL_BEST_VAL ):
            GLOBAL_BEST_VAL = tmp_val
            np.copyto( GLOBAL_BEST, position ) # syntax: np.copyto(dst, src)
        
        # particles has position, best_pos, velocity
        swarm.append( dict( zip(particle_params, [position, position, np.zeros(n)]) ) )
    
    print( ">> ", GLOBAL_BEST_VAL )
    print( "## ", GLOBAL_BEST )
    
    for _ in range(epochs):
        for i, particle in enumerate(swarm):
            
            # checking P_LOCAL_BEST
            if fitness_function( particle['POSITION'], wt, vals, W ) > fitness_function( particle['P_LOCAL_BEST'], wt, vals, W ):
                np.copyto( particle['P_LOCAL_BEST'], particle['POSITION'] )
            
            # checking GLOBAL_BEST
            if fitness_function( particle['P_LOCAL_BEST'], wt, vals, W ) > GLOBAL_BEST_VAL:
                np.copyto( GLOBAL_BEST, particle['P_LOCAL_BEST'] )
                GLOBAL_BEST_VAL = fitness_function( GLOBAL_BEST, wt, vals, W ) # update
            
            # random() generates values from range of <0,1>
            particle['VELOCITY'] += C1 * random() * (particle['P_LOCAL_BEST']-particle['POSITION']) + C2 * random() * (GLOBAL_BEST-particle['POSITION'])
            print(i, '*', particle['VELOCITY'])
            
            ro = np.random.randint( 2, dtype=np.int8, size=n ) # random binary array
            
            particle['POSITION'] = np.where( ro < sigmoid(particle['VELOCITY']), 1, 0 )
            
            # genetic mutations
            matation_arr = np.full(n, False, dtype=bool)
            matation_arr[ : randrange(n) ] = True
            np.random.shuffle( matation_arr )
            particle['POSITION'] = np.where( matation_arr, particle['P_LOCAL_BEST'], particle['POSITION'] )
            
            matation_arr = np.full(n, False, dtype=bool)
            matation_arr[ : randrange(n) ] = True
            np.random.shuffle( matation_arr )
            particle['POSITION'] = np.where( matation_arr, GLOBAL_BEST, particle['POSITION'] )
            
            print( ">> ", GLOBAL_BEST_VAL )
            print( "## ", GLOBAL_BEST )


solve_pso_knapsack( W, wt, val, 5, 3 )
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
            ti.append(solve_pso_knapsack(W, wt, val, j, int( np.sqrt(k)) ) )
    res = {'size':sz,'time':ti}
    return res

#print(test_Knapsack())

