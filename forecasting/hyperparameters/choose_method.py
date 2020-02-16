




def choose_method(hprm, method = None):
    
    if method is not None:
        hprm['learning.method'] = method
    else:
        hprm['learning.method'] = 'regression_tree'
                                   # 'additive_features_model'  Main standard bivariate linear model
                                   # 'xgboost' 
                                   # 'random_forests' 
                                   # 'regression_tree' 
                                   # 'svr'
                                   # 'gam'        
                                   # 'self_reg'   Regression on the instantaneous loads - not feasible in practice to forecast the loads
                                   # 'sr'         Old linear model - not used anymore
                                   # 'ls'         Old linear model - not used anymore
                                   # 'autoskl'    AutoSKLearn
                                   # 'nn'         Neural Networks   
                                   # 'filter'     Model with latent variables - given up
                                   
    hprm['learning.independent_models'] = True
    hprm['learning.individual_designs'] = False

    return hprm




