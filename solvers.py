import numpy as np
import scipy.sparse as sp

vec         = lambda mat: np.reshape(mat, (mat.shape[0] * mat.shape[1]), order = 'F')
vec_triu    = lambda mat: mat[np.triu_indices(mat.shape[0])]
zero        = lambda n1,n2: sp.csr_matrix((n1,n2))

class NewtonKKT(object):
    """
        Newton-KKT solver class

        solves:
                
                ||A - T S||_F^2     s.t.    (T_i)^T T_j = 0     i =/= j
                                            BT = G
                                            Tu = v 
    """
    def __init__(self, X, S, A, U, r, u, v, periodic_flag = True, B = None, G = None):
        """
            Copy data and constraint matrices
        """
        self.X = np.copy(X)
        self.S = np.copy(S)
        self.A = np.copy(A)
        self.U = np.copy(U)
        self.r = r

        self.nx, self.nt = self.X.shape

        self.u = np.copy(u)
        self.v = np.copy(v)

        self.B = np.copy(B)
        self.G = np.copy(G)
        self.periodic_flag = periodic_flag

        self.build_constant_sparse_submatrices()

        if not self.periodic_flag:
            self.num_vars = self.r**2 + self.nG1 * self.nG2 + self.r + self.num_striu
        else:
            self.num_vars = self.r**2 + self.r + self.num_striu

        if not self.periodic_flag:
            self.nB1, self.nB2 = self.B.shape
            self.nG1, self.nG2 = self.G.shape

        

    def build_constant_sparse_submatrices(self):
        """
            Build all a priori known submatrices of KKT Jacobian:

            |C11+C11d     C12     C13    C14|
            |C21          0       0      0  |
            |C31          0       0      0  |
            |C41          0       0      0  |

            that is:
                        -C11
                        -C12
                        -C13

            and their transposes

        """
        self.SST    = self.S @ self.S.T 
        self.AST    = self.A @ self.S.T
        Ir          = sp.eye(self.r)
        self.C11    = sp.kron(2*self.SST, Ir)

        if not self.periodic_flag:
            self.C12 = sp.kron(Ir, self.B.T)

        self.C13 = sp.kron(self.u[:,None], Ir)

        self.striu      = np.triu_indices(self.r, k = 1)
        self.num_striu  = len(self.striu[0])

        self.eij_mat = sp.csr_matrix((self.r,0))
        for i in range(self.r):
            for j in range(i+1, self.r):
                ei      = sp.lil_matrix((self.r, 1))
                ej      = sp.lil_matrix((self.r, 1))

                ei[i,0] = 1
                ej[j,0] = 1

                self.eij_mat = sp.hstack((self.eij_mat, sp.kron(ei,ej.T) + sp.kron(ej, ei.T)))

        self.eij_mat = sp.csr_matrix(self.eij_mat)

    def build_C14(self, T):
        """
            Builds C14 by computing:
             
                            T @ [(e11 + e11), (e12 + e21), ... ] 
            
            and reshaping to

                            [vec(T(e11 + e11)), vec(T(e12 + e21)), ... ]
        """
        Teij    = sp.csr_matrix(T) @ self.eij_mat
        C14     = Teij.reshape((self.r**2, -1), order = 'F')
        return C14

    def build_C11d(self, lamb):
        """
            Builds C11 by setting:

                    A   = triu(lamb)
                    C11 = kron(A + A.T, Ir)
        """
        arr             = np.zeros((self.r, self.r))
        arr[self.striu] = lamb

        Ir      = sp.eye(self.r)
        C11d    = sp.kron(arr + arr.T, Ir)
        return C11d

    def build_KKT_Jacobian(self, T, lamb):
        """
            construct the KKT Jacobian:

                    |C11+C11d     C12     C13    C14|
                    |C21          0       0      0  |
                    |C31          0       0      0  |
                    |C41          0       0      0  |

        """ 
        C11d    = self.build_C11d(lamb)
        C14     = self.build_C14(T)

        if not self.periodic_flag:
            n1 = self.C12.shape[1]

        n2 = self.C13.shape[1]
        n3 = C14.shape[1]

        if not self.periodic_flag:
            C1      = sp.hstack((self.C11 + C11d,   self.C12,            self.C13,          C14)) 
            C2      = sp.hstack((self.C12.T,        zero(n1,n1),         zero(n1,n2),       zero(n1,n3)))         
            C3      = sp.hstack((self.C13.T,        zero(n2,n1),         zero(n2,n2),       zero(n2,n3)))        
            C4      = sp.hstack((C14.T,             zero(n3,n1),         zero(n3,n2),       zero(n3,n3)))

            C = sp.vstack((C1,C2,C3,C4))
        else:
            C1      = sp.hstack((self.C11 + C11d,   self.C13,          C14))    
            C3      = sp.hstack((self.C13.T,        zero(n2,n2),       zero(n2,n3)))        
            C4      = sp.hstack((C14.T,             zero(n3,n2),       zero(n3,n3)))

            C = sp.vstack((C1,C3,C4))

        return sp.csr_matrix(C)
    
    def Langrangian_gradient(self, T, lamb, Gamma, nu):
        """
            Compute the gradient of the Lagrangian
        """
        arr             = np.zeros((self.r,self.r))
        arr[self.striu] = lamb
        Lambda          = arr + arr.T

        if not self.periodic_flag:
            dLdT        = 2 * T @ self.SST - 2 * self.AST + self.B.T @ Gamma + nu[:,None] @ self.u[:,None].T + T @ Lambda
            dLdGamma    = self.B @ T - self.G
            dLdnu       = T @ self.U - self.v
            dLdlambda   = (T.T @ T)[self.striu]

            return np.concatenate((vec(dLdT), vec(dLdGamma), dLdnu, dLdlambda))
        else:
            dLdT        = 2 * T @ self.SST - 2 * self.AST + nu[:,None] @ self.u[:,None].T + T @ Lambda
            dLdnu       = T @ self.u - self.v
            dLdlambda   = (T.T @ T)[self.striu]

            return np.concatenate((vec(dLdT), dLdnu, dLdlambda))

    def get_dT(self, p):
        return np.reshape(p[:self.r**2], (self.r, self.r), order = 'F')
    
    def get_dGamma(self, p):
        n1 = self.r**2
        n2 = n1 + self.nG1 * self.nG2
        return np.reshape(p[n1:n2], (self.nG1, self.nG2), order = 'F')
    
    def get_dnu(self, p):
        if not self.periodic_flag:
            n2 = self.r**2 + self.nG1 * self.nG2
            n3 = n2 + self.r
            return np.reshape(p[n2:n3], (self.r), order = 'F')
        else:
            n2 = self.r**2
            n3 = n2 + self.r
            return np.reshape(p[n2:n3], (self.r), order = 'F')

    def get_dLambda(self, p):
        if not self.periodic_flag:
            n2 = self.r**2 + self.nG1 * self.nG2 + self.r
            n3 = n2 + self.num_striu
            return np.reshape(p[n2:n3], (self.num_striu), order = 'F')
        else:
            n2 = self.r**2 + self.r
            n3 = n2 + self.num_striu
            return np.reshape(p[n2:n3], (self.num_striu), order = 'F')

    def objective(self, T, lamb, Gamma, nu):
        return np.linalg.norm(self.Langrangian_gradient(T, lamb, Gamma, nu))**2

    def solve(self, T, maxits = 5, reltol = 1e-5, lamb = None, Gamma = None, nu = None):
        if lamb is None:
            lamb = np.zeros((self.num_striu))
        if Gamma is None:
            Gamma = np.zeros_like(self.G)
        if nu is None:
            nu = np.zeros((self.r))

        it = 0
        not_done = True

        damping_factor = 1
        prev_objective = self.objective(T, lamb, Gamma, nu)

        while it < maxits and not_done:
            C = self.build_KKT_Jacobian(T, lamb)
            g = self.Langrangian_gradient(T, lamb, Gamma, nu)
            
            Told        = np.copy(T)
            nuold       = np.copy(nu)
            lambold     = np.copy(lamb)
            Gammaold    = np.copy(Gamma)

            if it < 5:
                p = sp.linalg.lsqr(C, -g, damp = damping_factor)[0]
            else:
                p = sp.linalg.spsolve(C, -g)

            T       = Told + self.get_dT(p)
            nu      = nuold + self.get_dnu(p)
            lamb    = lambold + self.get_dLambda(p)
            if not self.periodic_flag:
                Gamma = Gammaold + self.get_dGamma(p)


            objective_change = self.objective(T, lamb, Gamma, nu) - prev_objective
            prev_objective += objective_change

            rel_diff_dT = np.abs(objective_change)
            if rel_diff_dT < reltol:
                not_done = False

            print(it, rel_diff_dT, prev_objective, damping_factor)

            it += 1
            
        return T
            


        
    



