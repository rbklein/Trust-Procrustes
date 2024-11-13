import numpy as np
import scipy.sparse as sp

from timeit import default_timer as timer

vec         = lambda mat: np.reshape(mat, (mat.shape[0] * mat.shape[1]), order = 'F')
vec_triu    = lambda mat: mat[np.triu_indices(mat.shape[0])]
zero        = lambda n1,n2: sp.csr_matrix((n1,n2))

class ByrdOmojokun(object):
    """
        Solve an optimization problem using Byrd and Omojokun's two step trust-region SQP method

        see: "On the implementation of an algorithm for large-scale equality constrained optimization"
    """
    def __init__(self, problem, trust_radius, maxits = 100):
        #set optimization problem to be solved
        self.problem = problem

        #set algorithm parameters
        self.trust_radius           = trust_radius
        self.VERTICAL_STEP_FACTOR   = 0.8
        self.maxits                 = maxits

    #solving routine
    def solve(self, T):
        #estimate initial Lagrange multipliers
        multipliers = self.Lagrange_estimate(T)

        it = 0
        while it < self.maxits:
            v_step  = self.vertical_step(T)
            h_step  = self.horizontal_step(T, multipliers, v_step)
            step    = h_step

            rho     = self.evaluate_success(T, multipliers, step)

            if rho > 0.5:
                T                   = problem.step(T, step)
                self.trust_radius   = np.minimum(2.0 * self.trust_radius, 1e10)
            else:
                self.trust_radius   = np.maximum(0.5 * np.linalg.norm(step), 1e-10)

            multipliers = self.Lagrange_estimate(T)

            it += 1
            print('it: ', it, '\t', 'KKT: ', np.linalg.norm(problem.grad_L(T, multipliers)), '\t',  'red: ', rho, '\t', 'rad: ', self.trust_radius, '\t', 'step: ', np.linalg.norm(step))
            
        return T
    

    #evaluate if proposed step is successfull
    def evaluate_success(self, T, multipliers, step):
        #compute quadratic model parameters
        H   = self.problem.Lagrangian_Hessian(T, multipliers)
        g   = self.problem.grad_f(T)
        fk  = self.problem.f(T)
        A   = self.problem.compute_cons_Jacobian(T)
        c   = self.problem.compute_constraints(T)

        pHp = np.inner(step, H @ step)
        gp  = np.inner(g, step)

        #compute penalty factor
        if pHp > 0:
            sigma = 1
        else:
            sigma = 0

        mu = 10 * (gp + sigma * pHp) / (0.5 * np.linalg.norm(c, 1))
        
        #compute quadratic model at next step
        quad_model_p    = 1/2 * pHp + gp + fk
        lin_cons_p      = A @ step + c

        #compute quadratic model after zero step
        quad_model_z    = fk
        lin_cons_z      = c
        
        #compute merit function values for quadratic models
        quad_merit_p    = quad_model_p + mu * np.linalg.norm(lin_cons_p)
        quad_merit_z    = quad_model_z + mu * np.linalg.norm(lin_cons_z)

        #compute actual model at next step
        T_next          = self.problem.step(T, step)
        f_p             = self.problem.f(T_next)
        c_p             = self.problem.compute_constraints(T_next)

        #compute actual model at current step
        f_z             = self.problem.f(T)
        c_z             = self.problem.compute_constraints(T)

        #compute merit functions for actual model
        merit_p         = f_p + mu * np.linalg.norm(c_p)
        merit_z         = f_z + mu * np.linalg.norm(c_z)

        if np.isclose(quad_merit_z - quad_merit_p, 0):
            den = 1e-16
        else:
            den = (quad_merit_z - quad_merit_p) 

        success_factor  = (merit_z - merit_p) / den
        return success_factor
        

    #Langrange multiplier estimator
    def Lagrange_estimate(self, T):
        """
            Estimate of the Lagrange multipliers by minimizing first order optimality condition:

                min ||grad_f + Jc.T * lambda||
        """
        #compute terms
        g = self.problem.grad_f(T)
        A = self.problem.compute_cons_Jacobian(T)

        #solve minimization
        multipliers = sp.linalg.lsqr(A.T, -g)[0]
        return multipliers

    #project vector into the null space of a matrix
    def null_space_projection(self, A, r, full_accuracy = False):
        """
            Project the vector r in the null space of the m x n matrix A where n > m and A^T has full column rank.
            The projector   

                    P_A r = (I - A^T (AA^T)^-1 A) r    

            is expressed as   
                    
                    P_A r = r - A^T v   with    AA^T v = A r

            these are the optimiality conditions for

                    min ||r - A^T v||^2

            which is solve iteratively for sparse A. This approach is known as the normal equations approach. See:
            "On the solutions of equality constrained quadratric programming problems arising in optimization" Gould, Hribar and Nocedal
        """
        #solve minimization with or without machine precision accuracy
        if full_accuracy:
            v = sp.linalg.lsqr(A.T, r, atol = 0, btol = 0, conlim = 0)[0]
        else:
            v = sp.linalg.lsqr(A.T, r)[0]

        #define projected value
        g = r - A.T @ v
        return g

    #solve the horizontal SQP step using a projected conjugate gradient solver
    def horizontal_step(self, T, multipliers, vert_step):
        """
            minimize

                grad_f.T step + 0.5 * step.T * Hessian * step

            subject to

                Jc * step - Jc * vert_step = 0
                ||step|| < D
        """
        #compute QP problem matrix and vector
        H = problem.Lagrangian_Hessian(T, multipliers)
        c = problem.grad_f(T)
        #H = self.problem.A
        #c = self.problem.b

        #compute QP linear constraint matrix and vector
        A = problem.compute_cons_Jacobian(T)
        b = - A @ vert_step
        #A = self.problem.C
        #b = -self.problem.d

        #compute horizontal trust-radius
        radius = self.trust_radius 

        #compute feasible initial point                                    
        #x = sp.linalg.lsqr(A, -b)[0]                                       #most efficient but need to check if lack of accuracy matters
        #x = sp.linalg.lsqr(A, -b, atol = 0, btol = 0, conlim = 0)[0]       #second most efficient and equally accurate as dense
        x = A.T @ sp.linalg.spsolve(A @ A.T, -b)                            #could be made more efficient using sparse QR decomposition of A.T (chatGPT)

        if np.linalg.norm(x) > radius:
            ValueError('No solution to horizontal step, trust-region too small for minimal norm solution')
            return x

        #compute initial values
        r = H @ x + c
        g = self.null_space_projection(A, r)
        p = -g

        #precomputed terms
        rt_g    = max(np.inner(r, g), 1e-16) 
        Hp      = H @ p

        #define iteration parameters
        it      = 0
        maxits  = A.shape[1] - A.shape[0]

        #projected CG loop
        while it < maxits:
            if rt_g < 1e-15:
                #print('tolerance satisfied')
                break

            alpha   = rt_g / np.inner(p, Hp)
            x_next  = x + alpha * p 

            #check if step is within trust-region
            if np.linalg.norm(x_next) >= radius:
                poly_a  = alpha**2 * np.inner(p,p)
                poly_b  = 2 * alpha * np.inner(x,p)
                poly_c  = np.inner(x,x)-radius**2
                theta   = (-poly_b + np.sqrt(poly_b**2 - 4 * poly_a * poly_c)) / (2 * poly_a)
                #print('radius passed')
                return x + theta * alpha * p
            else:
                x = x_next

            rp      = r + alpha * Hp
            gp      = self.null_space_projection(A, rp)
            beta    = np.inner(rp, gp) / rt_g
            p       = - gp + beta * p    
            g       = gp
            r       = rp 

            rt_g    = np.inner(r, g)
            Hp      = H @ p

            it += 1
            #print(it, rt_g)

        return x

    #solve the vertical least-squares constraint satisfication step using a dogleg method
    def vertical_step(self, T):

        #compute QP problem matrix and vector
        cons_Jacobian   = self.problem.compute_cons_Jacobian(T)
        ck              = problem.compute_constraints(T)
        g               = cons_Jacobian.T @ ck

        #compute Cauchy and least norm Newton steps
        v       = cons_Jacobian @ g
        cauchy  = - np.inner(g,g) / np.inner(v,v) * g 
        newton  = - cons_Jacobian.T @ sp.linalg.spsolve(cons_Jacobian @ cons_Jacobian.T, ck)

        #compute dogleg trust radius
        radius = self.VERTICAL_STEP_FACTOR * self.trust_radius

        #solve dogleg approximation
        cauchy_norm = np.linalg.norm(cauchy)
        if cauchy_norm >= radius:
            return radius / cauchy_norm * cauchy
        else:
            """
                solve:
                        ||p1 + (tau - 1) *(p2 - p1)||^2 = D^2
                        a * tau^2 + b * tau + c = 0 
            """
            dp      = newton - cauchy
            dpTdp   = dp.T @ dp
            p1Tdp   = cauchy.T @ dp

            c = dpTdp + cauchy_norm**2 - 2 * p1Tdp - radius**2
            b = 2 * p1Tdp - 2 * dpTdp
            a = dpTdp

            tau = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
            if tau <= 2:
                return cauchy + (tau - 1) * dp
            else:
                return newton


if __name__ == "__main__":
    
    TEST = False

    if TEST:
        import matplotlib.pyplot as plt
        import test_problem as test
        num_vars = 5
        num_cons = 2

        problem         = test.Test_Problem(num_vars, num_cons)
        trust_radius    = 1
        solver          = ByrdOmojokun(problem, trust_radius, 10)
        
        x_exact,lamb    = problem.exact_solution()
        x               = solver.solve(np.ones(num_vars))
        print(x, x_exact)
    else:
        import matplotlib.pyplot as plt
        import prepare_data as dat
        import procustes_solvers as pr
        import quadrature_points as qp
        import procrustes_problem as crus

        data        = np.load('data_u.npy')
        nx, nt      = data.shape

        r           = 40
        m           = 40
        P, indices  = qp.deim(data, m)

        X, A, S, U  = dat.generate_problem_matrices(data, indices, r)
        u,v,B,G     = dat.generate_constraint_matrices(U, (0,nx-1), (0,m-1))
        T           = pr.solve_non_normal_procrustes(A, S, 1e-4)

        problem = crus.Procrustes_Problem(X, S, A, U, u, v)

        trust_radius = 10
        solver = ByrdOmojokun(problem, trust_radius, 1000)

        T = solver.solve(T)

        print(np.max(np.abs(np.sum(U @ T, axis = 1))))

        plt.figure()
        plt.plot(U @ T)

        plt.figure()
        plt.imshow(T.T @ T, interpolation = None)

        plt.show()

        #v_step = solver.vertical_step(T)
        #h_step = solver.horizontal_step(T, np.ones(problem.num_cons), v_step)

        #print(np.linalg.norm(problem.compute_cons_Jacobian(T) @ (h_step - v_step), np.inf))
        #print(np.linalg.norm(h_step))
    
