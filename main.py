import numpy as np
import matplotlib.pyplot as plt

import prepare_data as dat
import solvers as sol
import procustes_solvers as pr
import quadrature_points as qp

data        = np.load('data_u.npy')

r           = 15
P, indices  = qp.deim(data, r)

X, A, S, U = dat.generate_problem_matrices(data, indices, r)
nx, nt = X.shape

u,v,B,G = dat.generate_constraint_matrices(U, (0,nx-1), (0,r-1))

T   = pr.solve_non_normal_procrustes(A, S, 1e-5)
M   = U @ T

plt.figure()
plt.imshow(M.T @ M, interpolation = None)
print(np.max(M @ np.ones(r)))

solver  = sol.NewtonKKT(X, S, A, U, r, u, v)
T       = solver.solve(T, 1000, 1e-10)

M = U @ T

plt.figure()
plt.imshow(M.T @ M, interpolation = None)

print(np.max(M @ np.ones(r)))

plt.figure()
plt.plot(U @ T)
plt.show()