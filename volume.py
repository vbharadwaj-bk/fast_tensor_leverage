import numpy as np
import numpy.linalg as la

# This is a prototype of the leveraged volume sampling algorithm given in
# https://arxiv.org/pdf/1802.06749.pdf. The procedure VolumeSample is taken from
# https://arxiv.org/pdf/1705.06908.pdf (Reverse iterative volume sampling).

def r_iter_volume_sample(X, s):
    '''
    X \in R^{n x d, s \in {d..n} 
    '''
    n, d = X.shape
    Z = la.pinv(X.T @ X)
    p = 1.0 - np.diag(X @ Z @ X.T).copy()    
    S_full = list(range(n))
    S = list(range(n))

    while len(S) > s:
        #p_comp = np.zeros(n)
        #Sdet = la.det(X[S, :].T @ X[S, :])
        #for i in S:
        #    removed_set = [j for j in S if j != i]
        #    p_comp[i] = la.det(X[removed_set, :].T @ X[removed_set, :]) / Sdet

        # Sample an index according to the distribution p
        i = np.random.choice(S_full, p=p / np.sum(p))
        S.remove(i)
        v = Z @ X[i, :].T / np.sqrt(p[i])

        for j in S:
            p[j] -= (X[j, :] @ v) ** 2

        p[i] = 0.0
        p[p < 0] = 0.0
        Z += np.outer(v, v)

    return S

def generate_histogram():
    # Construct an i.i.d. Gaussian matrix
    n = 100
    d = 10 
    nsamples = 1000

    X = np.random.randn(n, d) 
    full_determinant = np.sqrt(la.det(X.T @ X))
    volumes = []

    for _ in range(nsamples):
        S = r_iter_volume_sample(X, d)
        submatrix = X[S, :]
        volume = np.sqrt(la.det(submatrix.T @ submatrix))
        volumes.append(volume / full_determinant)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.hist(volumes, bins=100)

    fig.savefig('volume_histogram.png')

if __name__=='__main__':
    generate_histogram() 


