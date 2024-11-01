#import jax
#jax.config.update("jax_enable_x64", True)
#import jax.numpy as jnp
import numpy as jnp

def solve_procrustes(A, B):
    '''
        solve:
                X = argmin ||A - XB||_F    s.t.    X.T X = I
        
        using SVD 
    '''
    mat1, _, mat2 = jnp.linalg.svd(A @ B.T)
    return mat1 @ mat2

def solve_diagonal_left_orthogonally_weighted_procrustes(A, B, X):
    '''
        solve:
                d = argmin ||A - X diag(d) B||_F^2   with X.T X = I

        we use
                D/Dd(-2 tr(B.T diag(d) X.T A))  = -2 diag(BA.T X)
                D/Dd(B.T diag(d^2) B)           = 2 diag(BB.T) d 
    '''
    return jnp.diag(B @ A.T @ X) / jnp.diag(B @ B.T)

def solve_non_normal_procrustes(A, B, rel_tol = 1e-10, max_its = 100000):
    '''
        solve:
                X, d = argmin ||A - XB||_F      s.t.    X.T X = diag(d^2)
        
        using Everson's tandem method: ''Orthogonal, but not Orthonormal, Procrustes Problems''
    '''
    r = B.shape[0]

    d = jnp.ones(r)
    it = 0
    err = 1
    rhonew = 1000

    #precompute recurring constant terms
    ABT = A @ B.T
    diagBBT = jnp.diag(B @ B.T)
 
    while err > rel_tol and it < max_its:
        rho = rhonew

        #solve procrustes ||A - X diag(d) B||_F s.t. X^TX = I
        X1, _, X2 = jnp.linalg.svd(ABT @ jnp.diag(d))
        X = X1 @ X2

        #solve diagonal left orthogonal weighted procrustes ||A - X diag(d) B||_F with X^TX = I
        d = jnp.diag(X.T @ ABT) / diagBBT
        rhonew = jnp.linalg.norm(A - X @ jnp.diag(d) @ B, 'fro')**2

        err = jnp.abs(rhonew - rho) / jnp.abs(rho)

        it += 1
        if jnp.mod(it, 100) == 0:
            print(it, err, rhonew)
    
    return X @ jnp.diag(d)


