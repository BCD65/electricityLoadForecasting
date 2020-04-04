

"""
Script used to check consistency in the parameters
"""


#import numpy as np
import types
#from numbers   import Number
#from termcolor import colored

def check_hyperparameters(hprm):


    
    if hprm['sites.zone'] == 'all' and hprm['sites.aggregation'] == 'sum':
        assert hprm['stations.zone']        == 'all'
        assert hprm['stations.aggregation'] == 'weighted_mean'    
#    
#    """
#    Select variables
#    """
#    assert param['posts_id'] == param.get('posts_for_lag_id', param['posts_id']) # Otherwise, need to recheck parts of code that deal with posts_id
#
#
#    """
#    Check method                                         
#    """
#    if param['method'] in {'rf', 'sr', 'ls'}:
#        assert param['one_design'] + param['different_designs'] + param['shared_plus_ind'] == 1#+ param['dikt_1d_features'] 
#        assert param['sep_model'] or not param['different_designs']
#        
#        
#    """ 
#    Check qo
#    """
#    if param['method'] in {'sr', 'ls'}:
#        assert param['qo_prod'] + param['qo_min'] + param['qo_rbf2'] + param['qo_tps'] == 1
#
#    """
#    Check rf
#    """
#    if param['method'] == 'rf':
#        param['selected_variables'] = tuple(sorted({e for e in param['selected_variables'] if e != 'stamp'}))
#
#    """
#    Set approx_tf
#    """
    if hprm['learning.model'] in {'afm'}:
        ordered_index = hprm['afm.formula'].astype(str).sort_values(['coefficient', 'input']).index
        hprm['afm.formula'] = hprm['afm.formula'].loc[ordered_index]
        
        assert not set(hprm['afm.formula']['input'].unique()).difference(hprm['inputs.selection'])
        
#        if param['separation_var']:
#            #assert 0
#            param['tf_config_coef'] = {var:tuple(key 
#                                                 for key in sorted(param['tf_config_coef'][var])
#                                                 if (   key!=param['separation_var'][0]
#                                                     or (    '#' in key
#                                                         and key.split('#')[0]!=param['separation_var'][0] 
#                                                         and key.split('#')[1]!=param['separation_var'][0]
#                                                         ) 
#                                                     )
#                                                  )
#                                        for var in param['tf_config_coef']
#                                       }
##            param['tf_univariate'] = tuple(e 
##                                           for e in sorted(param['tf_univariate']) 
##                                           if e!=param['separation_var'][0]
##                                           )
##            param['tf_bivariate']  = tuple(e 
##                                           for e in sorted(param['tf_bivariate'])  
##                                           if (    e.split('#')[0]!=param['separation_var'][0] 
##                                               and e.split('#')[1]!=param['separation_var'][0]
##                                               )
##                                           )
#        for coef, list_keys in param['tf_config_coef'].items():
#            for key in list_keys:
#                for k in key.split('#'):
#                   assert k in param['selected_variables'], key
#    if param['method'] in {'approx_tf', }:
#        if np.sum(['lbfgs' in key for key, value in param['tf_config_coef'].items() if value]):
#            assert set(param['tf_config_coef'].keys()) == {'lbfgs_coef'}, param['tf_config_coef']
#            param.update({
#                          'tf_col_upd' : {}, 
#                          'tf_try_ws'  : False, 
#                          'tf_early_stop_test'      : False, 
#                          'tf_early_stop_ind_test'  : False, 
#                          'tf_early_small_decrease' : False, 
#                          })
#            for k, d in param['tf_pen'].items():
#                if 'lbfgs' in k:# in {'lbfgs_uni', 'lbfgs_bi'}:
#                    for a, b in d.items():
#                        if b not in {'', 'rsm'} and 'sm' not in b:
#                            print(colored(' '.join(['switching penalty for ', a , b, 'to rsm']), 'red', 'on_cyan'))
#                            d[a] = 'rsm'
#        for coef, list_keys in param['tf_config_coef'].items():
#            for cpl in list_keys:
#                for key in cpl.split('#'):
#                    assert key in param['selected_variables'], key
#        param['selected_variables'] = tuple(sorted([k 
#                                                    for k in param['selected_variables'] 
#                                                    if  k in [key for coef, list_keys in param['tf_config_coef'].items() for key in list_keys]
#                                                    or  k in [key for coef, list_bivkeys in param['tf_config_coef'].items() for bivkey in list_bivkeys for key in bivkey.split('#')]
#                                                             or param['separation_var'] and param['separation_var'][0] == k
#                                                    ])
#                                            )
#        #for c in param['tf_bivariate']:
#        #    for e in c.split('#'):
#        #        assert e in param['selected_variables'], (c, e)
#        for var, dikt in param['tf_pen'].items():
#            if var in param['tf_alpha']:
#                for key, pen in dikt.items():
#                    if pen in {'ncvxclasso', 'n2cvxclasso', 'row2sm'}:
#                        assert type(param['tf_alpha'][var][key]) == tuple
#                    else:
#                        assert isinstance(param['tf_alpha'].get(var,{}).get(key,0), Number) 
#        if 'Blr' in param['tf_config_coef']:
#            param['tf_active_set']   = False
#            for key in param['tf_col_upd'].keys():
#                param['tf_col_upd'][key] = False
#                assert not param['tf_col_upd'][key], ('pb col_upd', key)
#
#            
#        if type(param['stations_id']) == str and param['stations_id'] in {'mean', 'wmean'}:
#            #for key in param['selected_variables']:#['targetlag', 'nebu', 'meteo', 'meteolag', 'meteodif']:
#            for id_mask in ['tf_masks']:#, 'tf_maask2']:
#                for key in list(param[id_mask].keys()):
#                    if 'meteo' in key or 'nebu' in key:
#                        #print(colored(' '.join([' '*12, '{0:12.12}'.format(key), 'removed from', id_mask]), 'red'))
#                        del param[id_mask][key]            
#        if type(param['posts_id']) == str and param['posts_id'] in {'mean', 'sum'}:
#            for key in param['selected_variables']:#['targetlag', 'nebu', 'meteo', 'meteolag', 'meteodif']:
#                for id_mask in ['tf_masks']:#, 'tf_maask2']:
#                    if key in param[id_mask]:
#                        #print(colored(' '.join([' '*12, '{0:12.12}'.format(key), 'removed from', id_mask]), 'red'))
#                        del param[id_mask][key]
#            assert not param['tf_masks']
#            #assert not param['tf_maask2']
#            assert 'targetlag' not in param['tf_masks'] 
#            #assert 'targetlag' not in param['tf_maask2']
#            assert not set(param['tf_masks']) & {'nebu', 'meteo', 'meteolag', 'meteodif'}
#            #assert not set(param['tf_maask2']) & {'nebu', 'meteo', 'meteolag', 'meteodif'}
#        if param['stations_id'] in {'wmean', 'mean', 'sum'}:
#            assert not set(param['tf_masks']) & {'nebu', 'meteo', 'meteolag', 'meteodif'}
#            #assert not set(param['tf_maask2']) & {'nebu', 'meteo', 'meteolag', 'meteodif'}
#            
#        if 'target' in [key for coef, list_keys in param['tf_config_coef'].items() for key in list_keys]:
#            print(' '*20, colored('target in tf_univariate', 'red'))
#            assert len(param['tf_config_coef'])        == 1
#            assert len(param['tf_config_coef']['Bsp']) == 1
#            
#        if param['posts_id'] not in ('sum', 'mean') and param['zone'] != 'nat':
#            if 'Blr' not in param['tf_config_coef']:
#                if 'Bsp' in param['tf_config_coef']:
#                    pass # For low-rank experiments
#                else:
#                    assert param['tf_masks']
#                    #assert param['tf_maask2']
#        
#        # Posts_id
#        if type(param['posts_id']) == str and param['posts_id'] == 'sum':
#            for cat in sorted(param['tf_masks']):
#                if 'target' in cat:
#                    del param['tf_masks'][cat]
#
#            
#        # Stations_id    
#        if type(param['stations_id']) == str and param['stations_id'] in {'mean', 'wmean'}:
#            for cat in sorted(param['tf_masks']):
#                if 'meteo' in cat or 'nebu' in cat:
#                    del param['tf_masks'][cat]
#
#        else:
#            assert param['stations_id'] == 'all'
#            
#        if not param.get('tf_precompute_test', True):
#            assert not param['tf_early_stop_test']
#            assert not param['tf_early_stop_ind_test']
#            
#        if 'tf_orthogonalize' in param:
#            for k in param['tf_orthogonalize']:
#                assert tuple(sorted(k)) == k, k
#
#
#    if param['method'] == 'gam':
#        #assert param['posts_id'] in {'sum'}
#        param['selected_variables'] = tuple(sorted(param['selected_variables']))
#        for key in list(param['gam_dikt_uni'].keys()):
#            assert key  in param['selected_variables'], key
#            if param['separation_var'] and param['separation_var'][0] == key:
#                del param['gam_dikt_uni'][key]
#        for key1, key2 in list(param['gam_dikt_bi'].keys()):
#            assert key1 in param['selected_variables'], key1
#            assert key2 in param['selected_variables'], key2
#            if param['separation_var']:
#                if param['separation_var'][0] == key1 or param['separation_var'][0] == key2:
#                    del param['gam_dikt_bi'][key1,key2]
#                
#   
    for k, v in hprm.items():
        assert type(v) != types.GeneratorType, k     
#    
#    if param['separation_var']:
#        assert param['separation_var'][0] in param['selected_variables']
#        
#    for key in param['selected_variables']:
#        assert key in param['data_cat'], key
#    
#    assert param.get('selected_variables') == tuple(sorted(param.get('selected_variables')))
        
    return hprm