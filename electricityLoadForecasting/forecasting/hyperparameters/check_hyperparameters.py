
"""
Script used to check consistency in the parameters
"""

import types


def check_hyperparameters(hprm):
    
    if hprm['sites.zone'] == 'all' and hprm['sites.aggregation'] == 'sum':
        assert hprm['weather.zone']        == 'all', hprm['weather.zone']
        assert hprm['weather.aggregation'] == 'national_weighted_mean', hprm['weather.aggregation']   


    if hprm['learning.model'] in {'afm'}:
        assert hprm['afm.algorithm'] in ['L-BFGS', 'FirstOrder']
        hprm['afm.formula'].sort_index(inplace = True)
        set_diff =  {e
                     for inpts in hprm['afm.formula'].index.get_level_values('input').unique()
                     for e in ([inpts]
                               if type(inpts[0]) == str
                               else
                               inpts)
                     }.difference(hprm['inputs.selection'])
        assert not set_diff, set_diff
         


    if hprm['learning.model'] == 'gam':
        for inpt in hprm['gam.univariate_functions']:
            assert inpt in hprm['inputs.selection'], inpt
        for inpt1, inpt2 in hprm['gam.bivariate_functions']:
            assert inpt1 in hprm['inputs.selection'], inpt1
            assert inpt2 in hprm['inputs.selection'], inpt2

                  
    for k, v in hprm.items():
        assert type(v) != types.GeneratorType, k     

        
    return hprm