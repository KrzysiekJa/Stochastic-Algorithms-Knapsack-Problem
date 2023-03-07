'''
 0 / 1 Knapsack in simple:

Complexity Analysis:

Time Complexity: O(N*W). As redundant calculations of states are avoided.

Auxiliary Space: O(W) As we are using 1-D array instead of 2-D array.
'''

from decorator import bench
import numpy as np


@bench
def knapSack(W, wt, val, n):
    dp = [0 for i in range(W+1)]
    for i in range(1, n+1):
        for w in range(W, 0, -1):
            if wt[i-1] <= w:
                dp[w] = max(dp[w], dp[w-wt[i-1]]+val[i-1])
    return dp[W]


@bench
def knapSack_numpy(W, wt, val, n):
    dp = np.zeros(W+1, dtype=int)
    for i in range(1, n+1): 
        for w in range(W, 0, -1):
            if wt[i-1] <= w:
                dp[w] = max(dp[w], dp[w-wt[i-1]]+val[i-1])
    return dp[W]


@bench
def knapSack_numpy2(W, wt, val, n):
    dp = np.zeros(W+1)
    for i in range(1, n+1):
        dp[wt[i-1]:] = np.maximum(dp[wt[i-1]:], dp[:-wt[i-1]] + val[i-1])
        
    return int(dp[W])



#instance of the problem - test
#W = 6
#wt = [1,2,10,200,3,1,1]
#val = [2,5,6,1,100,1,100]
#knapSack( 10, [1,20,10,2,2], [5,100,10,100,1],5)


