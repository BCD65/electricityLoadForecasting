




def choose_model(hprm, model = None):
    
    if model is not None:
        hprm['learning.model'] = model
    else:
        hprm['learning.model'] = 'afm'
                                   # 'afm'  Main standard bivariate linear model
                                   # 'mars'
                                   # 'random_forests' 
                                   # 'regression_tree' 
                                   # 'svr'
                                   # 'xgboost' 
                                   # 'gam'

    # Learn independent models for the different values of an input (e.g. the hour)
    hprm['learning.model.separation.input'] = ()
    
    # Loop over the sites to learn the models independently
    hprm['learning.model.separation.sites'] = False
                                   
    # # Learn the benchmark models for the sites independently (duplicate)
    # hprm['learning.model.benchmarks.independent_models'] = True
    
    # # Customize the inputs for the benchmark models (duplicate)
    # hprm['learning.model.benchmarks.individual_inputs'] = False

    return hprm




