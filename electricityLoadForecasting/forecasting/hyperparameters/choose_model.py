




def choose_model(hprm, model = None):
    
    if model is not None:
        hprm['learning.model'] = model
    else:
        hprm['learning.model'] = 'random_forests'
                                   # 'additive_features_model'  Main standard bivariate linear model
                                   # 'xgboost' 
                                   # 'random_forests' 
                                   # 'regression_tree' 
                                   # 'svr'
                                   # 'gam'
                                   # 'nn'         Neural Networks   
                                   
    hprm['learning.independent_models'] = True
    hprm['learning.individual_designs'] = False

    return hprm




