import numpy as np 
from numpy.linalg import LinAlgError

def KL(p, q):
    """ 
    Compute the Kullback-Leibler divergence between two probability distributions.
    
    Parameters
    ----------
    p : array-like
        Probability distribution.
        
    q : array-like
        Probability distribution.
    
    Returns
    -------
    kl : float
        Kullback-Leibler divergence.
    """
    p = np.asarray(p)
    q = np.asarray(q)
    kl = np.sum(p * np.log(p / q))
    return kl

def E(P):
    """    
    Compute the entropy of P (matrix of size (n,m)).
    """
    n, m = P.shape
    E = 0
    P = np.asarray(P)
    for i in range(n):
        for j in range(m):
            E += - P[i,j] * (np.log(P[i,j]) - 1)
    return E

def h(r):
    """ 
    Compute the entropy of r.
    """
    r = np.asarray(r)
    h = - np.sum(r * np.log(r))
    return h

def Froebenius(A, B):
    """
    Compute the Frobenius dot product of matrix A and B
    """
    if A.shape != B.shape:
        raise ValueError("Matrices must have the same shape")
    A = np.asarray(A)
    B = np.asarray(B)
    return np.sum(A * B)

def is_positive_definite(A):
    """
    Check if the matrix A is positive definite.
    A matrix is positive definite if all its eigenvalues are positive.
    """
    try:
        eigenvalues = np.linalg.eigvals(A)
        return np.all(eigenvalues > 0)
    except LinAlgError:
        return False
    
def is_in_sigma(r, d):
    """
    Check if r is in the simplex of dimension d.
    """
    r = np.asarray(r)
    if np.sum(r) == d: 
        return True
    else:
        return False

def transportation_polytope(P, r, c):
    """ 
    Check if P is in the transportation polytope of r and c.
    """
    P = np.asarray(P)
    r = np.asarray(r)
    c = np.asarray(c)
    len_r = len(r)
    len_c = len(c)
    if P.shape != (len_r, len_c) or len_r != len_c:
        raise ValueError("P must have the same shape as r and c ; r and c must have the same length")
    else:
        d = len_r
        if is_positive_definite(P):
            if (P @ np.ones(d) == r) and (P.T @ np.ones(d) == c):
                return True
            else:
                return False
