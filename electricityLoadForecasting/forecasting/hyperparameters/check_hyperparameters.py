
import types


def check_hyperparameters(hprm):

    if hprm['sites.zone'] == 'all' and hprm['sites.aggregation'] == 'sum':
        assert hprm['weather.zone']        == 'all', hprm['weather.zone']
        assert hprm['weather.aggregation'] == 'national_weighted_mean', hprm['weather.aggregation']   


    if hprm['learning.model'] in {'afm'}:
        assert hprm['afm.algorithm'] in ['L-BFGS', 'FirstOrder']
        if (    hprm['afm.algorithm'] == 'L-BFGS'
            and set(hprm['afm.formula'].index.get_level_values('coefficient').unique()) != {'unconstrained'}
            ):
            raise ValueError('L-BFGS only implemented for unconstrained coefficients')

                  
    for k, v in hprm.items():
        assert type(v) != types.GeneratorType, k     

        
    return hprm