
import types


def check_hyperparameters(hprm):


    if hprm['sites.zone'] == 'all' and hprm['sites.aggregation'] == 'sum':
        assert hprm['weather.zone']        == 'all', hprm['weather.zone']
        assert hprm['weather.aggregation'] == 'national_weighted_mean', hprm['weather.aggregation']   


    if hprm['learning.model'] in {'afm'}:
        assert hprm['afm.algorithm'] in ['L-BFGS', 'FirstOrder']
            
            
    if hprm['learning.model'] in ['gam',
                                  'mars',
                                  ]:
        assert hprm['learning.model.separation.sites'] == True

                  
    for k, v in hprm.items():
        assert type(v) != types.GeneratorType, k     

        
    return hprm