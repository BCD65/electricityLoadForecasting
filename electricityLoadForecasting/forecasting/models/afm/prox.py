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

def prox_L1(M, alpha, coef_zero = None):
    raise NotImplementedError
#    if alpha == 0:
#        N = M
#    else:
#        N = np.multiply(M,(1 - alpha/(np.abs(M) + (M == 0))).clip(min = 0))
#    if EXTRA_CHECK:
#        min_prox_pb = 0.5*np.linalg.norm(M-N)**2 + alpha * np.abs(N).sum()
#        prox_pb_old = 0.5*np.linalg.norm(M-coef_zero)**2 + alpha * np.abs(coef_zero).sum()
#        print()
#        print('{0:30}'.format('0.5*np.linalg.norm(M)**2'), 0.5*np.linalg.norm(M)**2)
#        print('{0:30}'.format('alpha * np.abs(M).sum()'),  alpha * np.abs(M).sum())
#        print('{0:30}'.format('0.5*np.linalg.norm(M-N)**2'), 0.5*np.linalg.norm(M-N)**2)
#        print('{0:30}'.format('alpha * np.abs(N).sum()'),  alpha * np.abs(N).sum())
#        print()
#        print('{0:30}'.format('prox_pb_old'),    prox_pb_old)
#        print('{0:30}'.format('min_prox_pb : '), min_prox_pb)
#        if not min_prox_pb <= prox_pb_old + 1e-12:
#            import ipdb; ipdb.set_trace()
#    return N        


def prox_L2(M, alpha):
    raise NotImplementedError
#    return (1/(1+alpha))*M


def prox_enet(M, alpha, lamb):
    raise NotImplementedError
#    if alpha*lamb != 0:
#        M = np.multiply(M, (1 - alpha*lamb/(np.abs(M) + (M == 0))).clip(min = 0))
#        M = (1/(1 + alpha*(1 - lamb)))*M
#    return M   


def prox_col_lasso(U, mu):
    raise NotImplementedError
#    if mu != 0:
#        p, r = U.shape
#        for j in range(r):
#            c_norm = np.linalg.norm(U[:,j])
#            if c_norm > 0:
#                U[:,j] *= max(0, 1 - mu/c_norm)
#    return U 


def prox_row_lasso(U, mu):
    raise NotImplementedError
#    if mu != 0:
#        p, r = U.shape
#        for i in range(p):
#            r_norm = np.linalg.norm(U[i])
#            if r_norm > 0:
#                U[i] *= max(0, 1 - mu/r_norm)
#    return U
#
#

# Did not use the correct surrogate
def prox_ncvx_classo(U, mu, flag = False, coef_zero = None):
    raise NotImplementedError
#    alpha, beta, gamma = mu
#    assert gamma > 0
#    assert alpha > beta
#    p, r = U.shape
#    old_pen = 0
#    new_pen = 0
#    for j in range(r):
#        if alpha != 0 or beta != 0:
#            if type(U) not in {np.matrix, np.ndarray}:
#                import ipdb; ipdb.set_trace()
#            c_norm = np.linalg.norm(U[:,j])
#            if c_norm > 0:
#                pen_0 = min(alpha*c_norm, beta*c_norm + gamma)
#                old_pen += pen_0
#                # Future values of the proximal problems
#                pen1   = 0     +  0.5*c_norm**2 if c_norm < alpha else alpha*c_norm - 0.5*alpha**2
#                pen2   = gamma + (0.5*c_norm**2 if c_norm < beta  else beta *c_norm - 0.5*beta **2)
#                if flag:
#                    import ipdb; ipdb.set_trace()
#                if pen1 < pen2:
#                    c_norm_diff_sq = np.linalg.norm(U[:,j]*(1 - max(0, 1 - alpha/c_norm)))**2
#                    U[:,j]        *= max(0, 1 - alpha/c_norm)
#                    #c_norm_postbis    = np.linalg.norm(U[:,j])
#                    c_norm_post = c_norm*max(0, 1 - alpha/c_norm)
#                    #assert np.allclose(c_norm_post, c_norm_postbis)
#                    prox_pb1       = 0.5 * c_norm_diff_sq + alpha * c_norm_post 
#                    pen1_post      = alpha * c_norm_post 
#                    prox_pb2       = 0.5 * c_norm_diff_sq + beta  * c_norm_post + gamma
#                    pen2_post      = beta  * c_norm_post + gamma 
#                    assert pen1_post <= prox_pb1
#                    assert pen1_post <= pen_0
#                    assert prox_pb1  <= 0.5*c_norm**2
#                    assert np.allclose(prox_pb1, pen1)
#                    assert prox_pb1  <= prox_pb2
#                    assert pen2      <= prox_pb2
#                    assert pen2_post <= beta *c_norm + gamma
#                    new_pen += pen1_post
#                else:
#                    c_norm_diff_sq = np.linalg.norm(U[:,j]*(1 - max(0, 1 - beta/c_norm)))**2
#                    U[:,j]        *= max(0, 1 - beta /c_norm)
#                    #c_norm_postbis    = np.linalg.norm(U[:,j])
#                    c_norm_post = c_norm*max(0, 1 - beta/c_norm)
#                    #assert np.allclose(c_norm_post, c_norm_postbis)
#                    prox_pb1       = 0.5 * c_norm_diff_sq + alpha * c_norm_post 
#                    pen1_post      = alpha * c_norm_post 
#                    prox_pb2       = 0.5 * c_norm_diff_sq + beta  * c_norm_post + gamma
#                    pen2_post      = beta  * c_norm_post + gamma 
#                    assert pen2_post <= prox_pb2
#                    assert pen2_post <= pen_0
#                    assert prox_pb2  <= 0.5*c_norm**2
#                    assert np.allclose(prox_pb2, pen2)
#                    assert prox_pb2  <  prox_pb1
#                    assert pen1      <= prox_pb1
#                    assert pen1_post <= alpha*c_norm
#                    new_pen += pen2_post
##    print()
##    print(mu)
##    print('old_pen', old_pen)
##    print('new_pen', new_pen)
#    return U

def prox_ncvx_classo_v2(U, eta, mu, U_old = None, flag = False, coef_zero = None):
    raise NotImplementedError
#    if type(U) not in {np.matrix, np.ndarray}:
#        import ipdb; ipdb.set_trace()
#    p, r = U.shape
#    for j in range(U.shape[1]):
#        slope = mu[j]*eta
#        if slope != 0:#alpha != 0 or beta != 0:
#            c_norm = np.linalg.norm(U[:,j])
#            if c_norm > 0:
#                pen_0 = slope*c_norm #+ offset
#                
#                if EXTRA_CHECK:
#                    assert U.shape == coef_zero.shape
#                    M = U[:,j]*1
#                    N = M*max(0, 1 - slope/c_norm)
#                c_norm_diff_sq    = np.linalg.norm(U[:,j]*(1 - max(0, 1 - slope/c_norm)))**2
#                c_norm_diff_sqbis = (c_norm**2)*(1 - max(0, 1 - slope/c_norm))**2
#                assert np.allclose(c_norm_diff_sq, c_norm_diff_sqbis)
#                c_norm_post    = c_norm*max(0, 1 - slope/c_norm)
#                prox_pb        = 0.5 * c_norm_diff_sq + slope * c_norm_post #+offset
#                pen_post       = slope * c_norm_post #+ offset 
#                U[:,j]        *= max(0, 1 - slope/c_norm)
#                #prox_pb2       = 0.5 * c_norm_diff_sq + s  * c_norm_post + offset
#                #pen2_post      = beta  * c_norm_post + gamma 
#                assert pen_post <= prox_pb
#                assert pen_post <= pen_0
#                assert prox_pb  <= 0.5*c_norm**2 #+ offset
#                assert prox_pb  <= slope*c_norm  #+ offset
#                if EXTRA_CHECK:
#                    print('\nprox_pb', prox_pb, 
#                          'pen_post',  pen_post, 
#                          'c_norm_diff_sq', c_norm_diff_sq
#                          )
#                    min_prox_pb = 0.5*np.linalg.norm(M-N             )**2 + slope * np.linalg.norm(N)
#                    prox_pb_old = 0.5*np.linalg.norm(M-coef_zero[:,j])**2 + slope * np.linalg.norm(coef_zero[:,j])
#                    print()
#                    print('{0:40}'.format('0.5*np.linalg.norm(M)**2'), 0.5*np.linalg.norm(M)**2)
#                    print('{0:40}'.format('slope * np.linalg.norm(M)'),  slope * np.linalg.norm(M))
#                    print('{0:40}'.format('0.5*np.linalg.norm(M-N)**2'), 0.5*np.linalg.norm(M-N)**2)
#                    print('{0:40}'.format('slope * np.linalg.norm(N)'),  slope * np.linalg.norm(N))
#                    print('{0:40}'.format('0.5*np.linalg.norm(M-coef_zero[:,j])**2'), 0.5*np.linalg.norm(M-coef_zero[:,j])**2)
#                    print('{0:40}'.format('slope * np.linalg.norm(coef_zero[:,j])'),  slope * np.linalg.norm(coef_zero[:,j]))
#                    print()
#                    print('{0:40}'.format('prox_pb_old'),    prox_pb_old)
#                    print('{0:40}'.format('min_prox_pb : '), min_prox_pb)
#                    if not min_prox_pb <= prox_pb_old + 1e-12:
#                        import ipdb; ipdb.set_trace()
#    return U

def prox_thr_inf2(M, alpha):
    raise NotImplementedError
#    if alpha == 0:
#        return M
#    else:
#        p, r  = M.shape
#        for i in range(p):
#            if np.linalg.norm(M[i]) > 0:
#                M[i] *= int(np.linalg.norm(M[i])> alpha)
#        return M
#
def prox_TV(M, mu):
    raise NotImplementedError
#    if   M.ndim == 1 or M.shape[1] == 1:
#        return (tv1_1d(M - M.min(), mu) + M.min()).reshape(M.shape)
#    elif M.ndim == 2:
#        return np.array([tv1_1d(M[:,i] - M[:,i].min(), mu) + M[:,i].min() for i in range(M.shape[0])])
#    elif M.ndim == 3:
#        return np.array([[tv1_1d(M[:,j, k] - M[:,j, k].min(), mu) + M[:,j, k].min() for k in range(M.shape[2])] for j in range(M.shape[1])]).transpose(2, 0, 1)

 
def tf2_1d(z, mu, order = 2):
    raise NotImplementedError
#    if z.shape[0] <= 2:
#        return z
#    if order != 2:
#        assert 0
#    else:
#        y = z.reshape((-1, 1))
#        #print('hh')
#        n = y.shape[0]
#        #print('hh')
#        e = np.mat(np.ones((1, n)))
#        #print('hh')
#        D = scipy.sparse.spdiags(np.vstack((e, -2*e, e)), range(3), n-2, n)
#        #print('hh')
#        x = cvx.Variable(n)
#        #print('hh')
#        obj = cvx.Minimize(0.5 * cvx.sum_squares(z - x) + mu*cvx.norm(D * x, 1))
#        #print('hh')
#        prob = cvx.Problem(obj)
#        #print('kk')
#        try:
#            #print('ll')
#            prob.solve(verbose=False)
#            #print('aa')
#            if prob.status != cvx.OPTIMAL:
#                #print('gg')
#                raise Exception("Solver did not converge!")
#        except:
#            #print('uu')
#            prob.solve(verbose=True)
#            #print('bb') 
#            try:
#                if prob.status != cvx.OPTIMAL:
#                    #print('hh')
#                    raise Exception("Solver did not converge!")
#            except Exception as e:
#                print(e)
#                return z.reshape(-1)
##        print('hh')
##        print(np.array(x.value))
##        print(np.array(x.value).ndim)
##        print(np.array(x.value).shape)
##        assert np.array(x.value).ndim == 2
##        print('hh')
##        assert np.array(x.value).shape[1] == 1  
##        print('jj')
##        print(np.array(x.value))
#        return np.array(x.value).reshape(-1)
#    
# 
def prox_TF(M, mu):
    raise NotImplementedError
#    try:
#        if M.ndim == 0 or np.prod(M.shape) == 0:
#            return M
#        elif M.ndim == 1 or (M.ndim == 2 and M.shape[1] == 1):
#            return (tf2_1d(M - M.min(), mu) + M.min()).reshape(M.shape)
#        elif M.ndim == 2:
#            return np.array([tf2_1d(M[:,i] - M[:,i].min(), mu) + M[:,i].min() for i in range(M.shape[1])]).transpose(1, 0)
#        elif M.ndim == 3:
#            return np.array([
#                             [
#                              tf2_1d(M[:,j, k] - M[:,j, k].min(), mu) + M[:,j, k].min() \
#                              for k in range(M.shape[2])
#                              ] \
#                              for j in range(M.shape[1])
#                               ]).transpose(2, 0, 1)                             
#    except Exception as e:
#        print(e)
#        import ipdb; ipdb.set_trace()
    
