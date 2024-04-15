# Source code


```python
from scipy.sparse.linalg import eigsh, eigs
from scipy.sparse import bmat, diags
import numpy as np


def SpectralRadius(A):
    '''This function computes the spectral radius of an Hermitian matrix A
    Use: ρ = SpectralRadius(A)
    
    Input: 
        * A (scipy sparse array)
        
    Output:
        * ρ (float)
        
    '''

    ρ, _ = eigsh(A.astype(float), k = 1, which = 'LM')

    return ρ[0]


def SpectralRadiusNB(A):
    '''This function computes the spectral radius of the non-backtracking matrix
    Use: ρ = SpectralRadiusNB(A)
    
    Input:
        * A (scipy sparse array): graph adjacency matrix
        
    Outoput:
        * ρ (float)
    '''

    n, _ = A.shape
    D = diags(A@np.ones(n))
    Id = diags(np.ones(n))
    Bp = bmat([[A, -Id], [D-Id, None]])

    ρB, _  = eigs(Bp, k = 1, which = 'LM')
    return ρB[0].real
``` 