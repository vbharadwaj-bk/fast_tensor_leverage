def compute_exact_lowrank_soln(lhs, rhs):
        sigma_lhs, sigma_rhs = np.zeros(lhs.R, dtype=np.double), np.zeros(rhs.R, dtype=np.double)
        lhs.ten.get_sigma(sigma_lhs, j)
        rhs.ten.get_sigma(sigma_rhs, -1)

        g = chain_had_prod([lhs.U[i].T @ lhs.U[i] for i in range(N) if i != j])
        g_pinv = la.pinv(g) 

        elwise_prod = chain_had_prod([lhs.U[i].T @ rhs.U[i] for i in range(N) if i != j])
        elwise_prod *= np.outer(np.ones(lhs.R), sigma_rhs) 

        true_soln = rhs.U[j] @ elwise_prod.T @ g_pinv @ np.diag(sigma_lhs ** -1)