import numpy as np

def generate_problem_matrices(data, indices, r):
    """
        Generate the problem matrices used in the procrustes objective function:

            f(T) = ||A - T S||_F^2.

        Inserts a momentum-conserving mode into POD basis.

    """
    e       = np.ones((data.shape[0],1))
    e       = e / np.linalg.norm(e)
    data_e  = data - e @ e.T @ data
    data    = np.hstack((np.ones((data.shape[0],1)), data))

    X   = np.copy(data)
    S   = np.copy(data[indices,:])
    U   = np.linalg.svd(data_e, full_matrices = False)[0][:,:r-1]
    U   = np.hstack((e, U))
    A   = U.T @ X

    return X, A, S, U

def generate_constraint_matrices(U, boundary_inds = None, mode_inds = None):
    """
        Generate the constraint matrices and vectors

            u,v     s.t.    Tu = v
            B,G     s.t.    BT = G
    """
    nx, r = U.shape
    u = np.ones(r)
    v = U.T @ np.ones(nx)

    if boundary_inds is not None:
        B = U[boundary_inds,:]
        G = np.zeros((B.shape[0],r))
        for i, ind in enumerate(mode_inds):
            G[i,ind] = 1

        return u,v,B,G
    else:
        return u,v