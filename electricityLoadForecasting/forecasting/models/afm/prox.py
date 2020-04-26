import numpy as np
import scipy.sparse
try:
    import cvxpy as cvx
except:
    pass
try:
    from prox_tv import tv1_1d
except Exception:
    pass

#####################################
        
EXTRA_CHECK = 0


def prox_clipped_abs_deviation(U, eta, mu, U_old = None, flag = False, coef_zero = None):
    assert type(U) not in {np.matrix, np.ndarray}
    p, r = U.shape
    for j in range(U.shape[1]):
        slope = mu[j]*eta
        if slope != 0:
            c_norm = np.linalg.norm(U[:,j])
            if c_norm > 0:
                pen_0 = slope*c_norm
                if EXTRA_CHECK:
                    assert U.shape == coef_zero.shape
                    M = U[:,j]*1
                    N = M*max(0, 1 - slope/c_norm)
                c_norm_diff_sq    = np.linalg.norm(U[:,j]*(1 - max(0, 1 - slope/c_norm)))**2
                c_norm_diff_sqbis = (c_norm**2)*(1 - max(0, 1 - slope/c_norm))**2
                assert np.allclose(c_norm_diff_sq, c_norm_diff_sqbis)
                c_norm_post    = c_norm*max(0, 1 - slope/c_norm)
                prox_pb        = 0.5 * c_norm_diff_sq + slope * c_norm_post
                pen_post       = slope * c_norm_post
                U[:,j]        *= max(0, 1 - slope/c_norm)
                assert pen_post <= prox_pb
                assert pen_post <= pen_0
                assert prox_pb  <= 0.5*c_norm**2
                assert prox_pb  <= slope*c_norm
                if EXTRA_CHECK:
                    print('\nprox_pb', prox_pb, 
                          'pen_post',  pen_post, 
                          'c_norm_diff_sq', c_norm_diff_sq
                          )
                    min_prox_pb = 0.5*np.linalg.norm(M-N             )**2 + slope * np.linalg.norm(N)
                    prox_pb_old = 0.5*np.linalg.norm(M-coef_zero[:,j])**2 + slope * np.linalg.norm(coef_zero[:,j])
                    print()
                    print('{0:40}'.format('0.5*np.linalg.norm(M)**2'), 0.5*np.linalg.norm(M)**2)
                    print('{0:40}'.format('slope * np.linalg.norm(M)'),  slope * np.linalg.norm(M))
                    print('{0:40}'.format('0.5*np.linalg.norm(M-N)**2'), 0.5*np.linalg.norm(M-N)**2)
                    print('{0:40}'.format('slope * np.linalg.norm(N)'),  slope * np.linalg.norm(N))
                    print('{0:40}'.format('0.5*np.linalg.norm(M-coef_zero[:,j])**2'), 0.5*np.linalg.norm(M-coef_zero[:,j])**2)
                    print('{0:40}'.format('slope * np.linalg.norm(coef_zero[:,j])'),  slope * np.linalg.norm(coef_zero[:,j]))
                    print()
                    print('{0:40}'.format('prox_pb_old'),    prox_pb_old)
                    print('{0:40}'.format('min_prox_pb : '), min_prox_pb)
                    if not min_prox_pb <= prox_pb_old + 1e-12:
                        import ipdb; ipdb.set_trace()
    return U


def prox_col_group_lasso(U, mu):
    if mu != 0:
        p, r = U.shape
        for j in range(r):
            c_norm = np.linalg.norm(U[:,j])
            if c_norm > 0:
                U[:,j] *= max(0, 1 - mu/c_norm)
    return U 


def prox_elastic_net(M, eta, mu):
    alpha, theta = mu
    if alpha*eta*theta != 0:
        M = np.multiply(M, (1 - alpha*eta*theta/(np.abs(M) + (M == 0))).clip(min = 0))
        M = (1/(1 + alpha*eta*(1 - theta)))*M
    return M  


def prox_lasso(M,
            alpha,
            ):
    if alpha == 0:
        N = M
    else:
        N = np.multiply(M,(1 - alpha/(np.abs(M) + (M == 0))).clip(min = 0))
    return N        


# def prox_L2(M, alpha):
#     return (1/(1+alpha))*M 


def prox_row_group_lasso(U, mu):
    if mu != 0:
        p, r = U.shape
        for i in range(p):
            r_norm = np.linalg.norm(U[i])
            if r_norm > 0:
                U[i] *= max(0, 1 - mu/r_norm)
    return U


def prox_total_variation(M, mu):
    if   M.ndim == 1 or M.shape[1] == 1:
        return (tv1_1d(M - M.min(), mu) + M.min()).reshape(M.shape)
    elif M.ndim == 2:
        return np.array([tv1_1d(M[:,i] - M[:,i].min(), mu) + M[:,i].min() for i in range(M.shape[0])])
    elif M.ndim == 3:
        return np.array([[tv1_1d(M[:,j, k] - M[:,j, k].min(), mu) + M[:,j, k].min() for k in range(M.shape[2])] for j in range(M.shape[1])]).transpose(2, 0, 1)


def prox_trend_filtering(M, mu):
    if M.ndim == 0 or np.prod(M.shape) == 0:
        return M
    elif M.ndim == 1 or (M.ndim == 2 and M.shape[1] == 1):
        return (  tf2_1d(  M
                         - M.min(),
                         mu,
                         )
                + M.min()
                ).reshape(M.shape)
    elif M.ndim == 2:
        return np.array([(  tf2_1d(  M[:,i]
                                   - M[:,i].min(),
                                   mu,
                                   )
                          + M[:,i].min()
                          )
                         for i in range(M.shape[1])
                         ]).transpose(1, 0)
    elif M.ndim == 3:
        return np.array([[(tf2_1d(  M[:,j, k]
                                  - M[:,j, k].min(),
                                  mu,
                                  )
                           + M[:,j, k].min()
                           )
                          for k in range(M.shape[2])
                          ]
                         for j in range(M.shape[1])
                         ]).transpose(2, 0, 1)                             
    
    
def tf2_1d(z, mu, order = 2):
    raise NotImplementedError
    if z.shape[0] <= 2:
        return z
    if order != 2:
        assert 0
    else:
        y    = z.reshape((-1, 1))
        n    = y.shape[0]
        e    = np.mat(np.ones((1, n)))
        D    = scipy.sparse.spdiags(np.vstack((e,
                                               -2*e,
                                               e,
                                               )),
                                    range(3),
                                    n-2,
                                    n,
                                    )
        x    = cvx.Variable(n)
        obj  = cvx.Minimize(0.5 * cvx.sum_squares(z - x) + mu*cvx.norm(D * x, 1))
        prob = cvx.Problem(obj)
        try:
            prob.solve(verbose=False)
            if prob.status != cvx.OPTIMAL:
                raise Exception("Solver did not converge1!")
        except:
            prob.solve(verbose=True)
            try:
                if prob.status != cvx.OPTIMAL:
                    raise Exception("Solver did not converge2!")
            except Exception as e:
                print(e)
                return z.reshape(-1)
        return np.array(x.value).reshape(-1)