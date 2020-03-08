




def choose_model(hprm, model = None):
    
    if model is not None:
        hprm['learning.model'] = model
    else:
        hprm['learning.model'] = 'additive_features_model'
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




