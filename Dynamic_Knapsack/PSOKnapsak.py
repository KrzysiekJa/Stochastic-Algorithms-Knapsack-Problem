import random 
import matplotlib as mp 
import numpy as np
import sys

print(np.__version__)

W=5
wt = [1,2,10,200,3]
val = [2,5,6,1,100]

# W = maximum capacity
# wt = weight
# val = value
# n = size of the objects
# s = size of particles

def fitness_function(x,w,v,W):
    if(np.dot(x,w)<=W):
        return np.dot(x,v)
    else:
        return 0

def solve_pso_knapsack(W, wt, val, n,s):
    swarm = []
    glb_value = -1
    glb = []

    X=0
    PLBEST =1
    VELOCITY_Y =2

    
    C1 = 2.05
    C2 = 2.05


    #particles has x , p, velocity
    for i in range(0,s):
        pi = ([],[],[])

        # Random Generating the xi in the particles
        # Initializing the local minimum in the swarm
        # find the global solution inside the swarm
        temporary_i = []
        for j in range(0,n):
            temporary_i.append(random.randint(0,1))
        tmp = fitness_function(temporary_i,wt,val,W)
        if(tmp>glb_value):
            glb_value = tmp
            glb = temporary_i
        
        velocity_i = [0]*n
        swarm.append((temporary_i,temporary_i,velocity_i))
    
    sigmoid = lambda x :  1/(1 + np.exp(-x))

    print(swarm)
    print("\n\n")

    for it in range(0,3):
        for i in range (0,s):
            
            ro = []
            for j in range(0,n):
                ro.append(random.randint(0,1))

            for d in range(0,n-1):
                swarm[i][VELOCITY_Y][d] = C1* random.random()*(swarm[i][PLBEST][d]-swarm[i][X][d])+C2*random.random()*(glb[d]-swarm[i][X][d])
                print(swarm[i][VELOCITY_Y][d])
                # ro < s(v_i)
                # Change the logic of this part 

                
                if(ro[d]<sigmoid(swarm[i][VELOCITY_Y][d])):
                    swarm[i][X][d]=1
                else:
                    swarm[i][X][d]=0
    print(swarm)


solve_pso_knapsack(10,[1,20,10,2,2],[5,100,10,100,1],5,2)

