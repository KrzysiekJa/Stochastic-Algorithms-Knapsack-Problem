import random 
import matplotlib as mp 
import numpy as np
import sys
from decorator import bench


W=5
wt = [1,2,10,200,3]
val = [2,5,6,1,100]

# W = maximum capacity
# wt = weight
# val = value
# n = size of the objects
# s = size of particles

#introduce a flexible penality, exponential scaling, take the exponetial value instead the normal fitness it should grow faster
def fitness_function(x,w,v,W):
    if(np.dot(x,w)<=W):
        return np.dot(x,v)
    else:
        return 0

def random_binary_list(n):
    rand = []
    for j in range(0,n):
        rand.append(random.randint(0,1))
    return rand


def solve_pso_knapsack(W, wt, val, n, n_particles):
    epochs = 5
    swarm = []
    glb_value = -1
    glb = []

    X=0
    PLBEST =1
    VELOCITY_Y =2

    
    C1 = 2.05
    C2 = 2.05


    #particles has x , p, velocity
    for i in range(n_particles):
        pi = ([],[],[])

        # Random Generating the xi in the particles
        # Initializing the local minimum in the swarm
        # find the global solution inside the swarm
        temporary_i = random_binary_list(n)
        tmp = fitness_function(temporary_i,wt,val,W)
        if(tmp>glb_value):
            glb_value = tmp
            glb = temporary_i[:]
        
        velocity_i = [0]*n
        swarm.append((temporary_i,temporary_i,velocity_i))
    
    sigmoid = lambda x :  1/(1 + np.exp(-x))

    #print(swarm)
    #print("\n\n")
    
    #print( "## ", glb )
    #print( ">> ", fitness_function(glb,wt,val,W) )
    
    for it in range(epochs):
        for i in range(n_particles):
            
            # checking PLBEST
            if fitness_function(swarm[i][X],wt,val,W) > fitness_function(swarm[i][PLBEST],wt,val,W):
                for d in range(n):
                    swarm[i][PLBEST][d] = swarm[i][X][d]
            
            # checking glb
            if fitness_function(swarm[i][PLBEST],wt,val,W) > fitness_function(glb,wt,val,W):
                for d in range(n):
                    glb[d] = swarm[i][PLBEST][d]
                #print( "### ", glb )
                #print( ">>> ", fitness_function(glb,wt,val,W) )
            
            ro = random_binary_list(n)

            for d in range(n):
                swarm[i][VELOCITY_Y][d] += C1 * random.random() * (swarm[i][PLBEST][d]-swarm[i][X][d])  + C2 * random.random() * (glb[d]-swarm[i][X][d])
                #print(swarm[i][VELOCITY_Y][d])
                
                if(ro[d] < sigmoid(swarm[i][VELOCITY_Y][d])):
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
                swarm[i][X][k] = glb[k]
            
            #print( "## ", glb )
            #print( ">> ", fitness_function(glb,wt,val,W) )
    
    #print(swarm)


#solve_pso_knapsack(10,[1,20,10,2,2],[5,100,10,100,1],5,2)


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
                   
print(test_Knapsack())

