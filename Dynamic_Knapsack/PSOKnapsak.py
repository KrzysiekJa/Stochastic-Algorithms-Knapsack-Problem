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

def random_binary_list(n):
    rand = []
    for j in range(0,n):
        rand.append(random.randint(0,1))
    return rand


def solve_pso_knapsack(W, wt, val, n,s):
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
    for i in range(0,s):
        pi = ([],[],[])

        # Random Generating the xi in the particles
        # Initializing the local minimum in the swarm
        # find the global solution inside the swarm
        temporary_i = random_binary_list(n)
        tmp = fitness_function(temporary_i,wt,val,W)
        if(tmp>glb_value):
            glb_value = tmp
            glb = temporary_i[:] # !!!!! reference!
        
        velocity_i = [0]*n
        swarm.append((temporary_i,temporary_i,velocity_i))
    
    sigmoid = lambda x :  1/(1 + np.exp(-x))

    print(swarm)
    print("\n\n")
    
    print( "## ", glb )
    print( ">> ", fitness_function(glb,wt,val,W) )
    
    for it in range(epochs):
        for i in range(s):
            
            # checking PLBEST
            if fitness_function(swarm[i][X],wt,val,W) > fitness_function(swarm[i][PLBEST],wt,val,W):
                for d in range(n):
                    swarm[i][PLBEST][d] = swarm[i][X][d]
            
            # checking glb
            if fitness_function(swarm[i][PLBEST],wt,val,W) > fitness_function(glb,wt,val,W):
                for d in range(n):
                    glb[d] = swarm[i][PLBEST][d]
                print( "### ", glb )
                print( ">>> ", fitness_function(glb,wt,val,W) )
            
            ro = random_binary_list(n)

            for d in range(n):
                swarm[i][VELOCITY_Y][d] += C1 * random.random() * (swarm[i][PLBEST][d]-swarm[i][X][d])  + C2 * random.random() * (glb[d]-swarm[i][X][d])
                print(swarm[i][VELOCITY_Y][d])
                # ro < s(v_i)
                # Change the logic of this part 
                if(ro[d] < sigmoid(swarm[i][VELOCITY_Y][d])):
                    swarm[i][X][d]=1
                else:
                    swarm[i][X][d]=0
            
            # genetic mutations
            nmut = random.randrange(n)
            for j in range(nmut):
                k = random.randrange(n)
                swarm[i][X][k] = swarm[i][PLBEST][k]
            
            nmut = random.randrange(n)
            for j in range(nmut):
                k = random.randrange(n)
                swarm[i][X][k] = glb[k]
            
            print( "## ", glb )
            print( ">> ", fitness_function(glb,wt,val,W) )
    
    print(swarm)


solve_pso_knapsack(10,[1,20,10,2,2],[5,100,10,100,1],5,2)


