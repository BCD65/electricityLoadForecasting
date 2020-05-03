
import types


def check_hyperparameters(hprm):

    if hprm['sites.zone'] == 'all' and hprm['sites.aggregation'] == 'sum':
        assert hprm['weather.zone']        == 'all', hprm['weather.zone']
        assert hprm['weather.aggregation'] == 'national_weighted_mean', hprm['weather.aggregation']   


    if hprm['learning.model'] in {'afm'}:
        
        assert hprm['afm.algorithm'] in ['L-BFGS', 'FirstOrder']
        
        if hprm['afm.algorithm'] == 'L-BFGS':
            if set(hprm['afm.formula'].index.get_level_values('coefficient').unique()) != {'unconstrained'}:
                raise ValueError('L-BFGS only implemented for unconstrained coefficients')
            set_diff =  set(hprm['afm.formula']['regularization_func'].values).difference(['',
                                                                                           'ridge',
                                                                                           'smoothing_reg',
                                                                                           'block_smoothing_reg',
                                                                                           ])
            assert not set_diff, set_diff

        elif hprm['afm.algorithm'] == 'FirstOrder':
            set_diff =  set(hprm['afm.formula']['regularization_func'].values).difference(['',
                                                                                           'ridge',
                                                                                           'smoothing_reg',
                                                                                           'factor_smoothing_reg',
                                                                                           'clipped_abs_deviation',
                                                                                           'col_group_lasso',
                                                                                           'elastic_net',
                                                                                           'lasso',
                                                                                           'row_group_lasso',
                                                                                           'total_variation',
                                                                                           'trend_filtering',
                                                                                           ])
            assert not set_diff, set_diff
        else:
            raise ValueError('Descent algorithms should be FirstOrder or L-BFGS and is currently : {0}'.format(hprm['afm.algorithm']))
                  
    for k, v in hprm.items():
        assert type(v) != types.GeneratorType, k     

        
    return hprm