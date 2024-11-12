import numpy as np
import matplotlib.pyplot as plt

import prepare_data as dat
import procustes_solvers as pr
import quadrature_points as qp

data        = np.load('data_u.npy')

r           = 30
m           = 30
P, indices  = qp.deim(data, m)

X, A, S, U = dat.generate_problem_matrices(data, indices, r)
nx, nt = X.shape

#u,v,B,G = dat.generate_constraint_matrices(U, (0,nx-1), (0,m-1))

T   = pr.solve_non_normal_procrustes(A, S, 1e-7)
Mold = U @ T

plt.figure()
plt.plot(U @ T)

plt.figure()
plt.plot(Mold @ S[:,-1]) 
plt.plot(Mold @ np.linalg.pinv(Mold) @ X[:,-1])
plt.plot(X[:,-1], '--')

plt.show()