import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve as scipy_solve

vec         = lambda mat: np.reshape(mat, (mat.shape[0] * mat.shape[1]), order = 'F')
vec_triu    = lambda mat: mat[np.triu_indices(mat.shape[0])]
zero        = lambda n1,n2: sp.csr_matrix((n1,n2))

class Test_Problem(object):
    """
        A test problem to validate SQP optimization software implementations.

        Problem formulation (linearly constrained QP):

                x = min 1/2 * x.T @ A @ x + b.T @ x   s.t.    Cx = d
        
        A: n x n randomly generated positive definite matrix in sparse data format
        C: m x n randomly generated full rank matrix in sparse data format
    """
    def __init__(self, n, m):
        #generate objective function values
        A       = np.random.rand(n,n)
        self.A  = sp.csr_matrix(A.T @ A + 0.1 * np.eye(n))
        self.b  = np.random.rand(n)

        #generate constraint values
        C       = np.random.randn(m, n)
        self.d  = np.random.rand(m)
    
        # Check if matrix is full rank
        rank        = np.linalg.matrix_rank(C)
        target_rank = min(m, n)
    
        if rank == target_rank:
            self.C = sp.csr_matrix(C)  
        else:
            #if rank deficient set singular values to nonzero
            U, S, Vt    = np.linalg.svd(C, full_matrices=False)
            S           = np.linspace(1, 2, target_rank)  
            self.C      = sp.csr_matrix(U @ np.diag(S) @ Vt)
    
    #objective function
    def f(self, x):
        return 0.5 * np.inner(x, self.A @ x) + np.inner(self.b, x)

    #compute objective gradient
    def grad_f(self, x):
        return self.A @ x + self.b

    #compute the linear constraint violations
    def compute_constraints(self, x):
        return self.C @ x - self.d

    #compute constraint Jacobian
    def compute_cons_Jacobian(self, x):
        return self.C

    #Hessian of Lagrangian functional
    def Lagrangian_Hessian(self, x, multipliers):
        return self.A
    
if __name__ == "__main__":
    problem = Test_Problem(5, 2)
    
