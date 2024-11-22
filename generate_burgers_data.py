import matplotlib.pyplot as plt
import numpy as np

length = 1
nx = 500
dx = length / nx

T = 0.3
nt = 2000
dt = T / nt

u0 = lambda x: 1 + 0.5 * np.sin(2*np.pi/length * x)
#u0 = lambda x: np.sin(2*np.pi/length * x)


x = np.linspace(0.5 * dx, length - 0.5 * dx, nx)
uh = u0(x)


def s(u):
    return -np.log(u)

def eta(u):
    return - 1 / u

def psi(u):
    return u / 2

def f(u):
    return (u**2) / 2

def flux(ul,ur):
    return 1/2 * ul * ur 
'''
def s(u):
    return 0.5 * u**2

def eta(u):
    return u

def psi(u):
    return 1/6 * u**3

def f(u):
    return (u**2) / 2

def flux(ul,ur):
    return 1/6 * (ur**2 + ul**2 + ul*ur)
'''
    
def Fr(ur):
    r = len(ur)
    U1 = np.ones((r,1)) @ ur[None,:]
    U2 = ur[:,None] @ np.ones((1,r))
    return flux(U1,U2)

ones = np.ones(nx-1)

nu = 0.001
L = np.zeros((nx,nx))
L += np.diag(ones,1) + np.diag(ones,-1) - np.diag(2 * np.ones(nx))
L[0,-1] = 1
L[-1,0] = 1

D = np.zeros((nx,nx))
D += np.diag(ones,1) - np.diag(ones,-1) 
D[0,-1] = -1
D[-1,0] = 1

def df(uh):
    ur = np.roll(uh,-1)
    ul = np.roll(uh, 1)
    return flux(ur,uh) - flux(uh,ul)

def dudt(uh):
    return -1/dx * df(uh) + 1/dx**2 * nu * L @ uh

def RK4(uh):
    k1 = dudt(uh)
    k2 = dudt(uh + k1 * dt / 2)
    k3 = dudt(uh + k2 * dt / 2)
    k4 = dudt(uh + k3 * dt)
    return uh + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

if __name__ == '__main__':
    step_t = 0

    Data = np.zeros((nx, nt+1))
    Data[:,0] = uh

    Data_df = np.zeros((nx, nt+1))
    Data_df[:,0] = df(uh)

    Data_f = np.zeros((nx,nt+1))
    Data_f[:,0] = f(uh)

    while step_t < nt:
        uh = RK4(uh)
        step_t += 1

        if step_t % 10 == 0:
            print(step_t / nt)

        Data[:,step_t] = uh
        Data_df[:,step_t] = df(uh)
        Data_f[:,step_t] = f(uh)

    np.save('data_u', Data)
    np.save('data_df', Data_df)
    np.save('data_f', Data_f)

    plt.plot(x, uh)
    plt.show()

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


def OPF(d, ABT):
    U,_,V = np.linalg.svd(ABT @ np.diag(d), full_matrices=False)
    return U @ V

def solve_D(V, ABT, B, r):
    d = np.zeros(r)
    for i in range(r):
        d[i] = V[:,i].T @ ABT[:,i] / (B[i,:].T @ B[i,:])
    return d

def generalized_procrustes(its, A, B, r):
    d = np.ones(r)
    it = 0
    err = 1
    rhonew = 1000

    ABT = A @ B.T

    while err > 1e-5 and it < its:
        rho = rhonew
        V = OPF(d, ABT)
        d = solve_D(V, ABT, B, r)
        rhonew = np.linalg.norm(A - V @ np.diag(d) @ B, 'fro')**2

        err = np.abs(rhonew - rho) / np.abs(rho)

        it += 1
        if np.mod(it, 100) == 0:
            print(it, err, rhonew)
    
    return V @ np.diag(d)


def generate_basis(data, indices, r):
    A = data
    B = data[indices,:]

    M = generalized_procrustes(100000, A, B, r)
    return M

def columns(data):
    return (data[:, i] for i in range(data.shape[1]))

def compute_flux_matrix_data(data_u, skip = 50):
    data_F = np.empty((nx,0))
    count = 0
    for u in columns(data_u):
        data_F = np.hstack((data_F, Fr(u)[:,::skip]))
        count += 1
        if np.mod(count, 100) == 0:
            print(count)
    return data_F

from scipy.sparse import lil_matrix, identity

def create_sparse_block_matrix(n, m, p, q):
    matrix = lil_matrix((n * m, n * m))

    identity_block = identity(m, format='lil')

    row_start = p * m
    col_start = q * m

    matrix[row_start:row_start + m, col_start:col_start + m] = identity_block

    return matrix.tocsr()

def insert_boundary_indices(inds, nx):
    m = inds.shape[0]
    if 0 not in inds:
        inds[m-2] = 0
    if nx-1 not in inds:
        inds[m-1] = nx-1
    return inds