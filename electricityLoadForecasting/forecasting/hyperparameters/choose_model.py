




def choose_model(hprm, model = None):
    
    if model is not None:
        hprm['learning.model'] = model
    else:
        hprm['learning.model'] = 'gam'
                                   # 'additive_features_model'  Main standard bivariate linear model
                                   # 'mars'
                                   # 'random_forests' 
                                   # 'regression_tree' 
                                   # 'svr'
                                   # 'xgboost' 
                                   # 'gam'
                                   
    hprm['learning.independent_models'] = True
    hprm['learning.individual_designs'] = False

    return hprm




