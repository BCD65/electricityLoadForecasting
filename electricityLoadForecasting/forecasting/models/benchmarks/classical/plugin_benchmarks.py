

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree     import DecisionTreeRegressor
from sklearn.svm      import SVR

try:
    from pyearth import Earth # pyeart should be installed manually : conda install -c conda-forge sklearn-contrib-py-earth
except ModuleNotFoundError:
    pass
try:
    from xgboost import XGBRegressor # xgboost should be installed manually : conda install -c conda-forge xgboost
except ModuleNotFoundError:
    pass



def fit_and_predict(inputs_training, Y_training, inputs_validation, hprm, assignments = {}):
    assert type(hprm['learning.independent_models']) == bool
    assert type(hprm['learning.individual_designs']) == bool
    if not hprm['learning.independent_models']:
        Y_hat_training, Y_hat_validation, model = call_fitter(inputs_training, 
                                                              Y_training, 
                                                              inputs_validation, 
                                                              hprm,
                                                              )
        
        Y_hat_training   = pd.DataFrame(Y_hat_training, 
                                        index   = Y_training.index, 
                                        columns = Y_training.columns,
                                        )
        Y_hat_validation = pd.DataFrame(Y_hat_validation, 
                                        index   = inputs_validation.index, 
                                        columns = Y_training.columns,
                                        )
    else:
        Y_hat_training   = pd.DataFrame(0, 
                                        index   = Y_training.index, 
                                        columns = Y_training.columns,
                                        )
        Y_hat_validation = pd.DataFrame(0, 
                                        index   = inputs_validation.index, 
                                        columns = Y_training.columns,
                                        )
        model = {}
        if hprm['learning.individual_designs']:
            for ii, site_name in enumerate(Y_training.columns):
                print('\r{0}/{1}'.format(ii, Y_training.shape[1]), end = '\r')
                columns_to_keep = [
                                   (name_input, transformation, parameter, location)
                                   for (name_input, transformation, parameter, location) in inputs_training.columns
                                   if (   name_input not in assignments
                                       or location in assignments[name_input]
                                       )
                                   ]
                Y_hat_training[site_name], Y_hat_validation[site_name], model[site_name] = call_fitter(inputs_training[columns_to_keep],
                                                                                                       Y_training[site_name],
                                                                                                       inputs_validation[columns_to_keep],
                                                                                                       hprm,
                                                                                                       )
        else :
            for ii, site_name in enumerate(Y_training.columns):
                print('\r{0}/{1}'.format(ii, Y_training.shape[1]), end = '\r')
                Y_hat_training[site_name], Y_hat_validation[site_name], model[site_name] = call_fitter(inputs_training, 
                                                                                                       Y_training[site_name],
                                                                                                       inputs_validation,
                                                                                                       hprm,
                                                                                                       )
    return Y_hat_training, Y_hat_validation, model


def call_fitter(X_training,
                Y_training,
                X_validation,
                hprm,
                ):
    
    X_mean = X_training.mean(axis = 0)
    X_std  = X_training.std(axis = 0)
    X_training   = (X_training   - X_mean)/X_std
    X_validation = (X_validation - X_mean)/X_std
    
    method = hprm['learning.model']
    
    if Y_training.ndim == 2 and Y_training.shape[1] == 1 : 
        Y_training = Y_training[:,0]  

    if method in {'random_forests', 'regression_tree'}:
        pass
    elif method in {'xgboost', 'svr', 'mars'}:
        assert Y_training.ndim == 1
        
    if method == 'mars':
        model = Earth(verbose = hprm['mars.verbose'],
                      thresh  = hprm['mars.thresh'],
                      )
    elif   method == 'random_forests':
        model = RandomForestRegressor(n_estimators = hprm['random_forests.n_estimators'])
    elif method == 'regression_tree':
        model = DecisionTreeRegressor()
    elif method == 'svr':
        model = SVR(C       = hprm['svr.C'],
                    epsilon = hprm['svr.epsilon'],
                    )
    elif method == 'xgboost':
        model = XGBRegressor()
    else:
        raise ValueError

    model.fit(X_training.values,
              Y_training.values,
              )
    Y_hat_training   = model.predict(X_training.values)
    Y_hat_validation = model.predict(X_validation.values)
    return Y_hat_training, Y_hat_validation, model
            
            
    
