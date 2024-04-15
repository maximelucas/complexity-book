# Source code


```python
import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import bicg
import matplotlib as mpl
import matplotlib.cm as cm
import networkx as nx
import matplotlib.pyplot as plt


def GetSquareGrid(shape):
    '''This function generates a square grid with a given shape
    
    Use: A, mapper = GetSquareGrid(shape)
    
    Inputs:
        * shape (tuple): number of rows and number of columns
        
    Outputs:
        * A (scipy sparse array): graph adjacency matrix
        * mapper (dictionary): it maps the coordinate of each node of the grid to the corresponding adjacency matrix index
        
    '''

    height, width = shape
    n = height*width

    pos = []
    for i in range(height):
        for j in range(width):
            pos.append(tuple([i,j]))

    mapper = dict(zip(pos, np.arange(n)))

    EL = []

    for p in pos:
        x = mapper[p]    

        if p[1] < width-1:
            y = mapper[tuple([p[0], p[1]+1])]
            EL.append([x,y])
            
        if p[0] < height-1:
            y = mapper[tuple([p[0]+1, p[1]])]
            EL.append([x,y])
        
    EL = np.array(EL)

    A = csr_matrix((np.ones(len(EL)), (EL[:,0], EL[:,1])), shape = (n,n))
    A = A + A.T
    
    
    return A, mapper


def GraphDiffusion(A, u, α, dt, T):
    '''This function performs a diffusion of the signal u on the graph for T steps.
    
    Use: ut = GraphDiffusion(A, u, α, dt, T)
    
    Inputs:
        * A (scipy sparse matrix): input graph adjacency matrix
        * u (array): signal at time t = 0
        * α (float): diffusion coefficient
        * dt (float): time discretization
        * T (integer): number of time steps
        
    Output:
        * ut (dictionary of arrays): contains the value of u for each time t
        
    '''

    # calculate the degree vector and matrix
    n, _ = A.shape
    d = A@np.ones(n)
    D = diags(d)

    # get the Laplacian and identity matrices (for convenience)
    L = D - A
    Id = diags(np.ones(n))

    # initialize ut
    ut = [u]

    # run the simulation
    for t in range(T):
        u = (Id-α*dt*L)@u
        ut.append(u)

    return dict(zip(np.arange(T), ut))


def PlotDiffusion(G, ut, time_indeces, save = False): 
    '''This function plots the output of a diffusion process on a network'''

    # get the plotting positions
    pos = nx.spring_layout(G)
    S = np.array([[pos[x[0]][0], pos[x[1]][0]] for x in G.edges])
    T = np.array([[pos[x[0]][1], pos[x[1]][1]] for x in G.edges])
    X = np.array(list(pos.values()))

    norm = mpl.colors.Normalize(vmin = 0, vmax = 1)
    cmap = cm.cool
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    fig, ax = plt.subplots(1, len(time_indeces), figsize = (5*len(time_indeces), 5))

    for i, t in enumerate(time_indeces):
        ax[i].plot(S,T, color = 'k', linewidth = 0.1, alpha = 0.6)
        colors = [m.to_rgba(x) for x in ut[t]]
        ax[i].scatter(X[:,0], X[:,1], color = colors, edgecolors = 'k', s = 150)
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].set_title(f't = {t}', fontsize = 30)

    plt.tight_layout()
    if save:
        plt.savefig('Figures/diff.png', dpi = 400)

    plt.show();


def Tikhonov(y, idx, λ, L):
    '''This function performs Tikhonov regularization
    
    Use: y_reg = Tikhonov(y, idx, shape, λ, L)
    
    Inputs:
        * y (array): input signal
        * idx (boolean array): if idx[i] = True then i \in Q, otherwise idx[i] = False
        * λ (float): regularization parameter
        * L (scipy sparse array): Laplacian matrix
        
    Output:
        * y_reg (array): regularized signal
        
    '''
    
    n, _ = L.shape
    
    # build the matrix Q
    Q = np.zeros(n)
    Q[idx] = 1
    Q = diags(Q)
    
    # solve efficiently the inversion problem
    x_reg, _ = bicg(Q + λ*L, Q@y)
    
    return x_reg


def MapWithNaN(df ,x, q):
    try:
        return df.loc[x][q]
    except:
        return None
```
