import numpy as np
from numpy.linalg import qr, solve
from scipy.linalg import norm

def NSI(A, tol=1E-14, maxiter=5000):
    '''Obtain Eigendecomposition of matrix A via Normalized Simultaneous QR Iteration in k steps'''
    #Get Eigenvalues, Eigenvectors via QR Algorithm (Normalized Simultaneous Iteration)
    m = A.shape[0]
    Q = np.identity(m)
    residual = 10
    lprev = np.ones(m)
    ctr = 0
    while norm(residual) > tol:
        Q,R = qr(A@Q)
        lam = np.diagonal(Q.T @ A @ Q) #Rayleigh Quotient Matrix
        residual = norm(lprev - np.sort(lam))
        lprev = np.sort(lam)
        ctr += 1
        if ctr == maxiter: break
    #print(ctr)
        
    return(lam)


def LanczosTri(A):
    '''Tridiagonalize Matrix A via Lanczos Iterations'''
    
    #Check if A is symmetric
    #if((A.transpose() != A).any()):
    #    print("WARNING: Input matrix is not symmetric")
    n = A.shape[0]
    x = np.ones(n)                      #Random Initial Vector
    V = np.zeros((n,1))                 #Tridiagonalizing Matrix

    #Begin Lanczos Iteration
    q = x/norm(x)
    V[:,0] = q
    r = A @ q
    a1 = q.T @ r
    r = r - a1*q
    b1 = norm(r)
    ctr = 0
    #print("a1 = %.12f, b1 = %.12f"%(a1,b1))
    for j in range(2,n+1):
        v = q
        q = r/b1
        r = A @ q - b1*v
        a1 = q.T @ r
        r = r - a1*q
        b1 = norm(r)
        
        #Append new column vector at the end of V
        V = np.hstack((V,np.reshape(q,(n,1))))

        #Reorthogonalize all previous v's
        V = qr(V)[0]

        ctr+=1
        
        if b1 == 0: 
            print("WARNING: Lanczos ended due to b1 = 0")
            return V #Need to reorthonormalize
        
        #print(np.trace(V.T@V)/j)
    #Check if V is orthonormal
    #print("|V.T@V - I| = ")
    #print(np.abs((V.T@V)-np.eye(n)))
    #if((V.T@V != np.eye(n)).any()):
    #    print("WARNING: V.T @ V != I: Orthonormality of Transform Lost")
        
    #Tridiagonal matrix similar to A
    T = V.T @ A @ V
    
    return T