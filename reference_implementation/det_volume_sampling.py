import numpy as np
import numpy.linalg as la

# This is a prototype of the leveraged volume sampling algorithm given in
# https://arxiv.org/pdf/1802.06749.pdf. The procedure VolumeSample is taken from
# https://arxiv.org/pdf/1705.06908.pdf (Reverse iterative volume sampling).

def volume_sample(X, s):
    '''
    X \in R^{n x d, s \in {d..n} 
    '''
    n, d = X.shape
    Z = la.pinv(X.T @ X)
    p = np.diag(X @ Z @ X.T).copy()
    S = list(range(n))

    while len(S) > s:
        # Sample an index according to the distribution p
        i = np.random.choice(S, p=p / np.sum(p))
        S.remove(i)
        v = Z @ X[i, :].T / np.sqrt(p[i])

        for j in S:
            p[j] -= (X[j, :] @ v) ** 2

        Z += np.outer(v, v)

    return S

if __name__=='__main__':
    # Construct an i.i.d. Gaussian matrix
    n = 1000 
    d = 10 

    X = np.random.randn(n, d)
    S = volume_sample(X, d)
    submatrix = X[S, :]
    print(np.sqrt(np.det(submatrix.T @ submatrix)))
