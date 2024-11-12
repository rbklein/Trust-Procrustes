import numpy as np

def deim(data, m):
    V, _,_ = np.linalg.svd(data)
    V = V[:,:m]
    #m = V.shape[1]
    
    indices = np.zeros(m, dtype=int)
    indices[0] = np.argmax(np.abs(V[:, 0]))
    
    P = np.zeros((V.shape[0], m))
    P[indices[0], 0] = 1.0
    
    for i in range(1, m):
        V_selected = V[:,:i]
        c = np.linalg.solve(P[:, :i].T @ V_selected, P[:, :i].T @ V[:, i])
        
        r = V[:, i] - V_selected @ c
        
        indices[i] = np.argmax(np.abs(r))
        P[indices[i], i] = 1.0
    
    return P, indices

def nodal(data, m):
    nx, nt = data.shape

    U = np.linalg.svd(data, full_matrices=False)[0][:,:m]
    A = U.T @ data
    A = A.T

    data = data.T
    indices = []

    for i in range(m):
        correlations = np.sum(np.abs(data.T @ A)**2, axis = 1)
        correlations[indices] = 0
        i = np.argmax(correlations)
        indices.append(i)
        A = A - np.outer(data[:,i], data[:,i]) @ A / (np.linalg.norm(data[:,i])**2)
    
    return np.eye(nx)[:,indices], indices