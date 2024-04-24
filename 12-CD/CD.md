# Source code


```python
import numpy as np
import itertools
from scipy.sparse import csr_matrix
import networkx as nx
from scipy.sparse import diags, bmat
from scipy.sparse.linalg import eigsh
from scipy.io import mmread
from sklearn.cluster import KMeans


def DCSBM(C_matrix,c, label, theta):
    ''' Function that generates the adjacency matrix A with n nodes and k communities
    
    Use: A, label = adj(C_matrix,c, label, theta)
    
    Input:
        * C_matrix (array of size k x k) : affinity matrix of the network C
        * c (scalar) : average connectivity of the network
        * label (array of size n) : vector containing the label of each node
        * theta  (array of size n) : vector with the intrinsic probability connection of each node
    
    Output:
        * A (sparse matrix of size n x n) : symmetric adjacency matrix
        * label (array): label vector of the nodes in the giant component if giant = True
    '''

    # number of communities
    k = len(np.unique(label))
    fs = list()
    ss = list()

    n = len(theta)
    c_v = C_matrix[label].T # (k x n) matrix where we store the value of the affinity wrt a given label for each node
    first = np.random.choice(n,int(n*c),p = theta/n) # we choose the nodes that should get connected wp = theta_i/n: the number of times the node appears equals to the number of connection it will have

    for i in range(k): 
        v = theta*c_v[i]
        first_selected = first[label[first] == i] # among the nodes of first, select those with label i
        fs.append(first_selected.tolist())
        second_selected = np.random.choice(n,len(first_selected), p = v/np.sum(v)) # choose the nodes to connect to the first_selected
        ss.append(second_selected.tolist())

    fs = list(itertools.chain(*fs))
    ss = list(itertools.chain(*ss))

    fs = np.array(fs)
    ss  = np.array(ss)

    edge_list = np.column_stack((fs,ss)) # create the edge list from the connection defined earlier

    edge_list = np.unique(edge_list, axis = 0) # remove edges appearing more than once
    edge_list = edge_list[edge_list[:,0] > edge_list[:,1]] # keep only the edges such that A_{ij} = 1 and i > j

    A = csr_matrix((np.ones(len(edge_list[:,0])), (edge_list[:,0], edge_list[:,1])), shape=(n, n))
    A = A + A.transpose() # symmetrize the matrix

    return A, label


def ErdosRenyi(n, c):
    '''This function returns the sparse adjacency matrix of a Erdos-Renyi random graph
    
    Use: A = ErdosRenyi(n, c)
    
    Input:
        * n (int): number of nodes
        * c (float): expected average degree
        
    Output:
        * A (scipy sparse array): (n x n) sparse adjacency matrix
    '''

    A, _ = DCSBM(np.ones(([c])),c, np.zeros(n).astype(int), np.ones(n))
    return A


def ComputeModularity(A, label):
    '''This function computes the modularity of a partition on a network.
    
    Use: mod = ComputeModularity(A, label)
    
    Input:
        * A (scipy sparse array): graph adjacency matrix
        * label (array): label vector
        
    Output:
        * mod (float): modularity
        
    '''

    all_labels = np.unique(label)

    mod = 0
    m = np.sum(A)

    for a in all_labels:

        idx = label == a
        Σ_in = np.sum(A[idx][:,idx])
        Σ_tot= np.sum(A[idx])
        mod += Σ_in/m - (Σ_tot/m)**2

    return mod

def BuildBp(A):
    '''This function ccreates the 2nx2n matrix Bp sharing the spectrum with the non-backtracking matrix
    Use: Bp = BuildBp(A)
    
    Input:
        * A (scipy sparse array): graph adjacency matrix
        
    Outoput:
        * Bp (scipy sparse array)
    '''

    n, _ = A.shape
    D = diags(A@np.ones(n))
    Id = diags(np.ones(n))
    Bp = bmat([[A, -Id], [D-Id, None]])

    return Bp



def SpectralClustering(M, which, k):
    '''
    This algorithm performs spectral clustering

    Use: label = SpectralClustering(M, which, k)

    Input:
        * M (scipy sparse array): matrix used to perform SC
        * which (string): sets which eigenvalues to compute: 'LA' are the largest, 'SA' the smallest
        * k (int): number of communities

    Output:
        * label (array): estimated label vector
    '''

    γ, X = eigsh(M, k = k, which = which)
    if which == 'LA':
        idx = np.argsort(γ)[::-1]
    else:
        idx = np.argsort(γ)
    
    X = X[:,idx]
    X = X[:,1:]            
    label = KMeans(n_clusters = k).fit(X).labels_

    return label    
``` 