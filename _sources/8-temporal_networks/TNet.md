# Source code


Here you can find some useful functions used in the next notebook. You should copy the code down below and and save it in a file named `Tnet.py` inside the folder `src`.


```python
import numpy as np
import pandas as pd 
from scipy.sparse import csr_matrix, diags


def ComputeReachabilityVector(df, T):
    '''Given a temporal network in the form ijt contained in the dataframe df, this function computes the reachability of time-respecting walks of length T
    
    Use: reachability = ComputeReachabilityVector(df, T)
    
    Inputs:
        * df (pandas dataframe): temporal network in the form ijt
        * T (int): number of steps in the walk
        
    Outputs:
        * reachability (array): returns the T reachability values for each time in [1,T]
    '''

    # create a mapping between the nodes and integers
    all_nodes = np.unique(df[['i', 'j']])
    n = len(all_nodes)
    
    # shift the smallest time-stamp to 0
    df.t = df.t - df.t.min()

    # split the edge list according to the time stamp
    dft = [df[df.t == t] for t in range(T)]
    
    # create a list of snapshot adjacency matrices in which each node is connected to itself
    At = [diags(np.ones(n)) for t in range(T)]

    # compute the snapshot adjacency matrices
    for t in range(T):
        if len(dft[t]) > 0:
            A = csr_matrix((np.ones(len(dft[t])), (dft[t].i, dft[t].j)), shape = (n,n))
            At[t] += A + A.T
        
    # compute the reachability vector given the adjacency matrices list
    reachability = ReachabilityFromAdjacencyList(At)
    
    return reachability


def ReachabilityFromAdjacencyList(At):
    '''This function computes the reachability of time respecting paths given a sequence of adjacency matrices with self loops
    
    Use: reachability = ReachabilityFromAdjacencyList(At)
    
    Inputs:
        * At (list of sparse arrays): the entries of this list are the adjacency matrices of each snapshot
        
    Outputs:
        * reachability (array): returns the T reachability values for each time in [1,T], where T is the number of time frames
    
    '''
    
    # compute the reachability matrix R_{ij}(t) = 1  if j can be reached from i in t steps or fewer.
    R = [At[0]]
    T = len(At)
    for t in range(1, T):
        print(f'Progress: {int(100*t/T)}%', end = '\r')
        R.append((At[t]@R[-1]).sign())
        
    # compute the average of the reachability matrix
    reachability = np.array([r.mean() for r in R])
    
    return reachability


def ComputeIntereventStatistic(df):
    '''Given a contact network in the format ijtτ, this function computes the time elapsed between the end of an 
    interaction between two nodes (ij) and the beginning of a new one between the same nodes. 
    This is repeated for all nodes
    
    Use: intervals = ComputeIntereventStatistic(df)
    
    Inputs:
        * df (pandas dataframe): temporal network in the format ijtτ
        
    Outpus:
        * intervals (array): list of interevent values for all pairs'''
    
    # index the dataframe by the contact indeces
    df.set_index(['i', 'j'], inplace = True)
    all_pairs = list(df.index)
    intervals = []

    # for each pair, select only the entries of df involving that pair
    for i, pair in enumerate(all_pairs):
        print(f'Progress: {int(i/len(all_pairs)*100)}%', end = '\r')
        ddf = df.loc[pair]
        
        # compute and store the interevent duration
        if len(ddf) > 1:
            intervals.append((ddf.t + ddf.τ - np.roll(ddf.t, 1))[1:].values)
   
    return np.concatenate(intervals)

```