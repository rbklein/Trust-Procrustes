import numpy as np
import scipy.sparse as sp
import scipy.optimize as opt

from timeit import default_timer as timer

vec         = lambda mat: np.reshape(mat, (mat.shape[0] * mat.shape[1]), order = 'F')
mat         = lambda vec, r: np.reshape(vec, (r, r), order = 'F')
vec_triu    = lambda mat: mat[np.triu_indices(mat.shape[0])]
zero        = lambda n1,n2: sp.csr_matrix((n1,n2))

class AugmentedLagrangianLS(object):
    """
        Solve a nonlinear least squares problem using an augmented Lagrangian approach

        see L.Vandenberghe ECE133B (Spring 2023)
    """

    def __init__(self, problem, tol = 1e-4):
        #set problem
        self.problem = problem
        
        #penalty parameter and multiplier values
        self.mu             = 1
        self.multipliers    = np.zeros(self.problem.num_cons)

        #set solver parameters
        self.tol = tol

    #compute the residual vector of the augmented least squares problem
    def augmented_Lagrangian_residual(self, T):
        robj = self.problem.residual(T)
        rc   = np.sqrt(self.mu) * self.problem.compute_constraints(T) + 1/(2 * np.sqrt(self.mu)) * self.multipliers
        return np.concatenate((robj, rc))

    #compute the Jacobian of the augmented 
    def augmented_Lagrangian_Jacobian(self, T):
        Jr  = self.problem.residual_Jacobian(T)
        Jc  = self.problem.compute_cons_Jacobian(T)
        J   = sp.vstack((Jr, np.sqrt(self.mu) * Jc))
        return J

    #solve loop
    def solve(self, T):
        gk = self.problem.compute_constraints(T)
        it = 0
        while it < 50:
            #run nonlinear least squares solver for sparse systems with low accuracy
            v_T                 = opt.least_squares(self.augmented_Lagrangian_residual, vec(T), self.augmented_Lagrangian_Jacobian, method ='trf', tr_solver = 'lsmr', ftol = self.tol, xtol = self.tol, verbose = 2).x
            T                   = mat(v_T, self.problem.r)
            gkp                 = self.problem.compute_constraints(T)
            self.multipliers    = self.multipliers + 2 * self.mu * self.problem.compute_constraints(T)

            if np.linalg.norm(gkp) < 0.25 * np.linalg.norm(gk):
                self.mu = self.mu
            else:
                self.mu = 2 * self.mu

            gk = gkp
            it += 1
            print(it, '\t', 'obj: ', np.linalg.norm(self.problem.f(T)), '\t', 'cons: ', np.linalg.norm(gkp))
        
        return T
    

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import prepare_data as dat
    import procustes_solvers as pr
    import quadrature_points as qp
    import procrustes_problem as crus

    data        = np.load('data_u.npy')
    nx, nt      = data.shape

    #_, s, _ = np.linalg.svd(data, full_matrices=False)
    #plt.semilogy(s, 'o')
    #plt.show()

    r           = 15
    m           = r-1
    P, indices  = qp.deim(data, m)
    indices = np.concatenate((indices, np.array([450])))

    X, A, S, U  = dat.generate_problem_matrices(data, indices, r)
    u,v,B,G     = dat.generate_constraint_matrices(U, (0,nx-1), (0,m-1))
    T           = pr.solve_non_normal_procrustes(A, S, 1e-4)

    Q,R = np.linalg.qr(S.T)

    #Compress mesh and snapshot scalings out of problem using POD and QR
    problem         = crus.Procrustes_Problem(X, R.T, A @ Q, U, u, v)
    problem_wrapper = crus.Scipy_Procrustes(problem)

    solver = AugmentedLagrangianLS(problem_wrapper)
    t1 = timer()
    T      = solver.solve(T)
    t2 = timer()

    print('time: ', t2 - t1)

    v_T = vec(T)
    T   = mat(v_T, r)

    print(np.max(np.abs(np.sum(U @ T, axis = 1))))

    plt.figure()
    plt.plot(U @ T)

    plt.figure()
    plt.imshow(T.T @ T, interpolation = None)

    plt.figure()
    plt.semilogy(np.diagonal(T.T @ T), 'o')

    plt.figure()
    plt.plot(U @ T @ S[:,-1])
    plt.plot(X[:,-1])
    plt.plot(indices, np.zeros_like(indices), '+')

    plt.show()





