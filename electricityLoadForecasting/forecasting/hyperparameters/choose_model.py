




def choose_model(hprm, model = None):
    
    if model is not None:
        hprm['learning.model'] = model
    else:
        hprm['learning.model'] = 'gam'
                                   # 'afm'  Main standard bivariate linear model
                                   # 'mars'
                                   # 'random_forests' 
                                   # 'regression_tree' 
                                   # 'svr'
                                   # 'xgboost' 
                                   # 'gam'

    # Learn independent models for the different values of an input (e.g. the hour)
    hprm['learning.model.separation.input'] = () # () # (('hour', '', ''),24,)
    
    # Loop over the sites to learn the models independently
    hprm['learning.model.separation.sites'] = False

    return hprm




