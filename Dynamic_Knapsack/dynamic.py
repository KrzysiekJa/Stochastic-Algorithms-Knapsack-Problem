'''
 0 / 1 Knapsack in Python in simple

Complexity Analysis:

Time Complexity: O(N*W). As redundant calculations of states are avoided.

Auxiliary Space: O(W) As we are using 1-D array instead of 2-D array.
'''

from decorator import bench

@bench
def knapSack(W, wt, val, n):
    dp = [0 for i in range(W+1)]  # Making the dp array
 
    for i in range(1, n+1):  # taking first i elements
        for w in range(W, 0, -1):  # starting from back,so that we also have data of
                                # previous computation when taking i-1 items
            if wt[i-1] <= w:
                # finding the maximum value
                dp[w] = max(dp[w], dp[w-wt[i-1]]+val[i-1])
    
    return dp[W]  # returning the maximum value of knapsack
 
 
# driver code
val = [60, 100, 120 ]
wt = [10, 20, 30 ]
W = 50
n = len(val)

#print("Sec : "+ str(knapSack(W, wt, val, n)))


import random

def test_Knapsack():
    val = []
    wt = []
    W = 50
    
    for i in range(1,8):
        k = 10**i
        for j in range(k,k*10,k*10):
            val = []
            wt = []
            for t in range(0,j):
                val.append(random.randint(0,j))
                wt.append(random.randint(0,j)%W)
            #print(val)
            #print(wt)
            print("Sec : "+ str(knapSack(W, wt, val, n)))
            print("\n\n")

         
test_Knapsack()  



   


    
