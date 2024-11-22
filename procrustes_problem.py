import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve as scipy_solve

vec         = lambda mat: np.reshape(mat, (mat.shape[0] * mat.shape[1]), order = 'F')
mat         = lambda vec, r: np.reshape(vec, (r, r), order = 'F')
vec_triu    = lambda mat: mat[np.triu_indices(mat.shape[0])]
zero        = lambda n1,n2: sp.csr_matrix((n1,n2))

class Procrustes_Problem(object):
    """
        Optimization problem given by:
                
                ||A - T S||_F^2     s.t.    (T_i)^T T_j = 0     i =/= j
                                            BT = G
                                            Tu = v 
    """
    def __init__(self, X, S, A, U, u, v, periodic_flag = True, B = None, G = None):
        """
            Copy data and constraint matrices
        """

        #copy matrices
        self.X = np.copy(X)
        self.S = np.copy(S)
        self.A = np.copy(A)
        self.U = np.copy(U)

        #Deduce shape parameters
        self.r, _           = self.A.shape
        self.m, _           = self.S.shape
        self.nx, self.nt    = self.X.shape

        #copy unit constraint values
        self.u = np.copy(u)
        self.v = np.copy(v)
        self.num_unit_cons = len(self.v)

        #copy boundary interpolation constraint values
        self.B = np.copy(B)
        self.G = np.copy(G)
        self.periodic_flag = periodic_flag
        if not self.periodic_flag:
            self.B1, self.B2 = self.B.shape
            self.G1, self.G2 = self.G.shape
            self.num_boundary_cons = self.G1 * self.G2
        else:
            self.num_boundary_cons = 0

        #precompute frequently used products
        self.SST = self.S @ self.S.T
        self.AST = self.A @ self.S.T
        self.AAT = self.A @ self.A.T
        if not self.periodic_flag:
            self.BTB = self.B.T @ self.B
            self.BTG = self.B.T @ self.G
        else:
            self.BTB = 0
            self.BTG = 0
        self.uuT = np.outer(self.u, self.u)
        self.vuT = np.outer(self.v, self.u)

        #determine orthogonality constraints
        self.triu_inds      = np.triu_indices(self.r, 1)
        self.num_orth_cons  = len(self.triu_inds[0])

        #determine number of optimization parameters, constraint and total variables (Lagrange multipliers and parameters)
        self.num_cons   = self.num_orth_cons + self.num_unit_cons + self.num_boundary_cons
        self.num_params = self.r * self.m
        self.num_vars   = self.num_params + self.num_cons 

        #prepare constant-valued constraint Jacobians
        self.init_orth_constraint()
        self.compute_unit_Jacobian()
        if not self.periodic_flag:
            self.compute_boundary_Jacobian() 



    #compute objective function
    def f(self, T):
        obj = np.trace(self.AAT - 2 * T.T @ self.AST + T.T @ T @ self.SST) 
        return obj

    #compute objective gradient
    def grad_f(self, T):
        g = 2 * (T @ self.SST - self.AST)
        return vec(g)

    #compute vectorized residual value of least squares problem
    def residual(self, T):
        r = self.A - T @ self.S
        return vec(r)

    #compute Jacobian of vectorized residual
    def residual_Jacobian(self, T):
        Ir  = sp.eye(self.r)
        J   = -sp.kron(self.S.T, Ir)
        return J

    #compute boundary constraint
    def boundary_cons(self, T):
        #check if boundary is not periodic
        assert not self.periodic_flag

        cb = self.B @ T - self.G
        return vec(cb)
    
    #compute sparse boundary constraint Jacobian:
    def compute_boundary_Jacobian(self):
        Ir = sp.eye(self.r)
        self.boundary_Jacobian = sp.kron(Ir, self.B)

    #compute unit constraint
    def unit_cons(self, T):
        cu = T @ self.u - self.v
        return cu

    #compute sparse unit constraint Jacobian
    def compute_unit_Jacobian(self):
        Ir = sp.eye(self.r)
        self.unit_Jacobian = sp.kron(self.u[:,None].T, Ir)
    
    #compute orthogonality constraint as a vector
    def orth_cons(self, T):
        co = (T.T @ T)[self.triu_inds]
        return co

    #initialize matrix used in orthogonality constraint computations
    def init_orth_constraint(self):
        self.eij_mat = sp.csr_matrix((self.r,0))
        for i in range(self.r):
            for j in range(i+1, self.r):
                ei      = sp.lil_matrix((self.r, 1))
                ej      = sp.lil_matrix((self.r, 1))

                ei[i,0] = 1
                ej[j,0] = 1

                self.eij_mat = sp.hstack((self.eij_mat, sp.kron(ei,ej.T) + sp.kron(ej, ei.T)))

        self.eij_mat = sp.csr_matrix(self.eij_mat)

    #compute orthogonality constraint Jacobian
    def compute_orth_Jacobian(self, T):
        Teij    = sp.csr_matrix(T) @ self.eij_mat
        mat     = Teij.reshape((self.r**2, -1), order = 'F').T
        return mat
    
    #compute constraint values
    def compute_constraints(self, T):
        orth = self.orth_cons(T)
        unit = self.unit_cons(T)
        if not self.periodic_flag:
            cons_vec = np.concatenate((self.boundary_cons(T), unit, orth))
        else:
            cons_vec = np.concatenate((unit,orth))
        return cons_vec

    #compute sparse constraint Jacobian of vectorized system
    def compute_cons_Jacobian(self, T):
        orth_Jacobian = self.compute_orth_Jacobian(T)
        if not self.periodic_flag:
            constraint_Jacobian = sp.vstack((self.boundary_Jacobian, self.unit_Jacobian, orth_Jacobian))
        else:
            constraint_Jacobian = sp.vstack((self.unit_Jacobian, orth_Jacobian))

        return constraint_Jacobian

    #compute Lagrangian
    def Lagrangian(self, T, multipliers):
        Gamma, nu, lamb = self.separate_multipliers(multipliers)

        if not self.periodic_flag:
            bc = np.trace(Gamma.T @ self.boundary_cons(T))  
        else:
            bc = 0

        uc = nu.T @ self.unit_cons(T)
        oc = lamb.T @ self.orth_cons(T)

        L = self.f(T) + bc + uc + oc
        return L
    
    #compute derivative of Lagrangian with respect to T
    def dLdT(self, T, multipliers):
        Gamma, nu, lamb = self.separate_multipliers(multipliers)

        dcodT                   = np.zeros((self.r, self.r))
        dcodT[self.triu_inds]   = lamb

        if not self.periodic_flag:
            dL = mat(self.grad_f(T), self.r) + self.B.T @ Gamma + np.outer(nu, self.u) + T @ (dcodT + dcodT.T)
        else:
            dL = mat(self.grad_f(T), self.r) + np.outer(nu, self.u) + T @ (dcodT + dcodT.T)

        return dL

    #compute gradient of Langrangian as a vector
    def grad_L(self, T, multipliers):
        dLdT        = vec(self.dLdT(T, multipliers))
        dLdnu       = self.unit_cons(T)
        dLdlamb     = self.orth_cons(T)

        if not self.periodic_flag:
            dLdGamma    = vec(self.boundary_cons(T))
            return np.concatenate((dLdT, dLdGamma, dLdnu, dLdlamb))
        else:
            return np.concatenate((dLdT, dLdnu, dLdlamb))

    #compute upper_triangle(lambda) + lower_triangle(lambda) 
    def compute_lambda_matrix(self, lamb):
        L   = np.zeros((self.r, self.r))
        L[self.triu_inds]   = lamb
        L                   = L + L.T
        return L

    #compute Lagrangian Hessian as a sparse matrix
    def Lagrangian_Hessian(self, T, multipliers):
        _, _, lamb = self.separate_multipliers(multipliers)

        Ir  = sp.eye(self.r)
        L   = self.compute_lambda_matrix(lamb)
        return sp.kron((2 * self.SST + L), Ir)

    #compute KKT system as a sparse matrix
    def KKT_mat(self, T, multipliers):
        HL                  = self.Lagrangian_Hessian(T, multipliers)
        constraint_Jacobian = self.compute_cons_Jacobian(T)
        
        KKT = sp.bmat([[HL,                     constraint_Jacobian.T],
                       [constraint_Jacobian,    None]])
        return KKT

    #solve Lagrangian Hessian linear system
    def solve_Hessian(self, T, multipliers, rhs):
        _, _, lamb  = self.separate_multipliers(multipliers)
        rhs_matrix  = np.reshape(rhs, (self.r, self.r), order = 'F')
        mat         = 2 * self.SST + self.compute_lambda_matrix(lamb)
        sol         = scipy_solve(mat, rhs_matrix, assume_a='sym')
        sol         = vec(sol)
        return sol

    #extract Gamma, Lambda, nu from an array of Lagrange multipliers
    def separate_multipliers(self, multipliers):
        if not self.periodic_flag:
            Gamma   = np.reshape(multipliers[:self.num_boundary_cons], (self.G1, self.G2), order = 'F')
            nu      = multipliers[self.num_boundary_cons:self.num_boundary_cons+self.num_unit_cons]
            lamb    = multipliers[self.num_boundary_cons+self.num_unit_cons:self.num_boundary_cons+self.num_unit_cons+self.num_orth_cons]
        else:
            Gamma   = 0
            nu      = multipliers[:self.num_unit_cons]
            lamb    = multipliers[self.num_unit_cons:]
        return Gamma, nu, lamb
    
    
    #process step
    def step(self, T, step):
        return T + np.reshape(step, (self.r, self.r), order = 'F') 


class Scipy_Procrustes(object):
    """
        Wrap a Procrustes problem object to accept vectorized inputs
    """
    def __init__(self, problem):
        self.prob       = problem
        self.num_cons   = problem.num_cons
        self.r          = problem.r

    def f(self, v_T):
        T   = mat(v_T, self.prob.r)
        obj = self.prob.f(T)
        return obj 

    def grad_f(self, v_T):
        T = mat(v_T, self.prob.r)
        g = self.prob.grad_f(T)
        return g
    
    def hess_f(self, v_T):
        Ir  = sp.eye(self.prob.r)
        return sp.kron((2 * self.prob.SST), Ir)
    
    def residual(self, v_T):
        T = mat(v_T, self.prob.r)
        r = self.prob.residual(T)
        return r
    
    def residual_Jacobian(self, v_T):
        T = mat(v_T, self.prob.r)
        J = self.prob.residual_Jacobian(T)
        return J

    def compute_constraints(self, v_T):
        T = mat(v_T, self.prob.r)
        c = self.prob.compute_constraints(T)
        return c 
    
    def compute_cons_Jacobian(self, v_T):
        T = mat(v_T, self.prob.r)
        J = self.prob.compute_cons_Jacobian(T)
        return J


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import prepare_data as dat
    import procustes_solvers as pr
    import quadrature_points as qp

    data        = np.load('data_u.npy')
    nx, nt      = data.shape

    r           = 20
    m           = 20
    P, indices  = qp.deim(data, m)

    X, A, S, U  = dat.generate_problem_matrices(data, indices, r)
    u,v,B,G     = dat.generate_constraint_matrices(U, (0,nx-1), (0,m-1))
    T           = pr.solve_non_normal_procrustes(A, S, 1e-7)

    problem = Procrustes_Problem(X, S, A, U, u, v)
    sol = problem.solve_Hessian(T, np.ones(problem.num_cons), np.ones(r**2))

    d = problem.JTJ_diagonal(T, 1)

    JTJ = np.zeros((r**2,r**2))
    for i in range(r**2):
        ei      = np.zeros(r**2)
        ei[i]   = 1
        JTJ[:,i] = problem.r_JTJp(T, ei) + problem.c_JTJp(T, ei)

    print(d - np.diagonal(JTJ))

    plt.figure()
    plt.plot(sol)

    plt.figure()
    plt.spy(problem.KKT_mat(T, np.ones(problem.num_cons)))
    plt.show()




"""
Code for solving Augmented Lagrangian by a normal equation approach. Code is
saved but approach not taking due to ill-conditioning of the normal equations

as class methods:
    
def compute_JTJ(self, T):
        JTJ = np.zeros((self.num_params, self.num_params))
        for i in range(self.num_params):
            ei      = np.zeros(self.num_params)
            ei[i]   = 1
            JTJ[:,i] = self.r_JTJp(T, ei) + self.c_JTJp(T, ei)
        return JTJ

    #diagonal values of JTJ for scaling trust-region methods
    def JTJ_diagonal(self, T, mu = 1):
        d = self.r_JTJ.diagonal() + mu * self.c_JTJ.diagonal()
        d += mu * sp.linalg.norm(self.compute_orth_Jacobian(T), 2, axis = 0)**2
        return d

    #compute constant part of residual Jacobian-Jacobian product
    def init_residual_JacJac_product(self):
        Ir          = sp.eye(self.r)
        self.r_JTJ  = sp.kron(self.SST, Ir)

    #compute constant part of constraint Jacobian-Jacobian product
    def init_constraint_JacJac_product(self):
        Ir          = sp.eye(self.r)
        self.c_JTJ  = sp.kron(np.outer(self.u,self.u), Ir) 
        if not self.periodic_flag:
            self.c_JTJ += sp.kron(Ir, self.BTB) 
    
    #compute product residual Jacobian-Jacobian-vector product
    def r_JTJp(self, T, p):
        JTJp = self.r_JTJ @ p
        return JTJp

    #compute product constraint Jacobian-Jacobian-vector product
    def c_JTJp(self, T, p):
        #compute product with orthogonality constraint JacJacVec product using subsequent sparse Jacobian multiplications
        Jc      = self.compute_orth_Jacobian(T)
        Jcp     = Jc @ p
        JTJp    = self.c_JTJ @ p + Jc.T @ Jcp
        return JTJp

    #compute a preconditioner method for use with the augmented Langrangian least-squares method
    def init_ALLS_preconditioner(self, penalty_parameter):
        JTJ             = self.r_JTJ + penalty_parameter * self.c_JTJ
        
        #preconditioner object with solve method
        return sp.linalg.splu(JTJ)
    
    #compute residual Jacobian product
    def r_JTb(self, T):
        JTb = -vec(self.AST - T @ self.SST)
        return JTb
    
    #compute constraint Jacobian product
    def c_JTb(self, T):
        c_JTb = vec(T @ self.uuT - self.vuT)
        if not self.periodic_flag:
            c_JTb += vec(self.BTB @ T - self.BTG)
        c_JTb += self.compute_orth_Jacobian(T).T @ (T.T @ T)[self.triu_inds]
        return c_JTb

    #computed constraint Jacobian product of augmented Lagrangian penalty
    def ALLS_JTb(self, T, multipliers):
        Gamma, nu, lamb = self.separate_multipliers(multipliers)

        JTb = 0.5 * vec(np.outer(nu, self.u))
        if not self.periodic_flag:
            JTb += 0.5 * vec(self.B.T @ Gamma)
        JTb += 0.5 * self.compute_orth_Jacobian(T).T @ lamb

        return JTb
"""