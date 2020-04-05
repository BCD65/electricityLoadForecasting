


import pandas as pd



"""
Default hyperparameters for additive features model
"""

              
#############################################################################
                     
#                          selected variables                             ###

#############################################################################

#all_selected_variables = (
#                           'dado',
#                           'dbdo',
#                           'dl',
#                           'do',
#                           'h',
#                           'meteo',
#                           'meteolag',
#                           'meteomax',
#                           'meteomin', 
#                           'nebu',
#                           'ones',
#                           'stamp',
#                           'targetlag',
#                           'wh',
#                           'xmas',
#                           'yd',
#                           #'meteodif',
#                           #'meteosmo', 
#                           #'sbrk',
#                           #'wd',
#                           )


             
###############################################################################
                     
###                       Stopping Criteria                                 ###

###############################################################################
              
stopping_criteria = {
                     ('Eco2mix.administrative_regions',
                      None,
                      ) : {
                               'afm.algorithm.lbfgs.maxfun'          : 1e8,
                               'afm.algorithm.lbfgs.maxiter'         : 1e8,                             
                               'afm.algorithm.lbfgs.pgtol'           : 1e-6,
                               'afm.algorithm.lbfgs.tol'             : 1e-12,
                               },
                     ('Eco2mix.France',
                      None,
                      ) : {
                               'afm.algorithm.lbfgs.tol'             : 1e-12,
                               'afm.algorithm.lbfgs.pgtol'           : 1e-6,
                               'afm.algorithm.lbfgs.maxfun'          : 1e8,
                               'afm.algorithm.lbfgs.maxiter'         : 1e8,                             
                               },
#                     'full' : {
#                               'tf_tol_lbfgs'             : 1e-9,
#                               'tf_pgtol_lbfgs'           : 1e-5,
#                               'tf_maxfun_lbfgs'          : 1e8,
#                               'tf_maxiter_lbfgs'         : 1e8,                             
#                               }
                     }
#stopping_criteria.update({
#                          'e2m'       : cp.deepcopy(stopping_criteria['nat']),
#                          'rteReg'    : cp.deepcopy(stopping_criteria['nat']),
#                          'admReg'    : cp.deepcopy(stopping_criteria['nat']),
#                          'districts' : cp.deepcopy(stopping_criteria['nat']),
#                          })
                     
                     
             
     
              
###############################################################################
                     
###                             dikt formula                                ###

###############################################################################                        

dikt_formula = {
                ###
                ('Eco2mix.France',
                 None,
                 ) : {
                      'tmp'   : pd.DataFrame([
                                              # Univariate features
                                              ('B', ('target',        'lag',        pd.DateOffset(hours = 24)), 4,    'r2sm', 1e-4),
                                              ('B', ('temperature',   '',           ''),                        16,   'r2sm', 1e-4),
                                              ('B', ('temperature',   'difference', pd.DateOffset(hours = 24)), 16,   'r2sm', 1e-2),
                                              ('B', ('temperature',   'maximum',    '1D'),                      16,   'r2sm', 1e-2),
                                              ('B', ('temperature',   'minimum',    '1D'),                      16,   'r2sm', 1e-1),
                                              ('B', ('temperature',   'smoothing',  0.99),                      16,   'r2sm', 1e-1),
                                              ('B', ('timestamp',     '',           ''),                        'p1', 'rsm',  1e-3),
                                              ('B', ('week_hour',     '',           ''),                        168,  'rsm',  1e-8),
                                              # Bivariate features
                                              
                                              ],
                                             columns = ['coefficient',
                                                        'input',
                                                        'nb_intervals',
                                                        'regularization_func',
                                                        'regularization_coef',
                                                        ],
                                             ).set_index(['coefficient', 'input'])
                      },
                ###
                ('Eco2mix.administrative_regions',
                 None,
                 ) : {
                      'tmp'   : pd.DataFrame([
                                              ('B', ('target',        'lag',        pd.DateOffset(hours = 24)), 4,    'r2sm', 1e-4),
                                              ('B', ('temperature',   '',           ''),                        16,   'r2sm', 1e-4),
                                              ('B', ('temperature',   'difference', pd.DateOffset(hours = 24)), 16,   'r2sm', 1e-2),
                                              ('B', ('temperature',   'maximum',    '1D'),                      16,   'r2sm', 1e-2),
                                              ('B', ('temperature',   'minimum',    '1D'),                      16,   'r2sm', 1e-1),
                                              ('B', ('temperature',   'smoothing',  0.99),                      16,   'r2sm', 1e-1),
                                              ('B', ('timestamp',     '',           ''),                        'p1', 'rsm',  1e-3),
                                              ('B', ('week_hour',     '',           ''),                        168,  'rsm',  1e-8),
                                              ],
                                             columns = ['coefficient',
                                                        'input',
                                                        'nb_intervals',
                                                        'regularization_func',
                                                        'regularization_coef',
                                                        ],
                                             ).set_index(['coefficient', 'input'])
                      },
                }
                                                                                        

####################################################################################
                                  
                                  

#                      'univariate'   : {
#                                        'B' : tuple([
#                                                       'meteo',
#                                                       'meteolag',
#                                                       'meteomax',
#                                                       'meteomin',
#                                                       'ones',
#                                                       'stamp',
#                                                       'targetlag',
#                                                       'wh',
#                                                       'xmas',
#                                                       'yd',
#                                                       'do', 
#                                                       'dado', 
#                                                       'dbdo', 
#                                                       'dl', 
#                                                       'nebu',
#                                                       ]),
#                                             },
#                           ###
#                           'lowrank' :      {
#                                             'Blr' : tuple([
#                                                            'wh',
#                                                            'yd',
#                                                            'do#wh',
#                                                            'dado#wh',
#                                                            'dbdo#wh',
#                                                            'wh#yd',
#                                                            ]),
#                                             'Bsp' : tuple([
#                                                            'meteo',
#                                                            'meteolag',
#                                                            'meteomax',
#                                                            'meteomin',
#                                                            'ones',
#                                                            'stamp',
#                                                            'targetlag',
#                                                            'xmas',
#                                                            'dl#nebu',
#                                                            'targetlag#wh',
#                                                            'meteo#yd',
#                                                            ]),
#                                             },
#                            ###
#                            'lbfgs'       : {
#                                             'lbfgs_coef' : tuple([
#                                                                  'meteo',
#                                                                  'meteolag',
#                                                                  'meteomax',
#                                                                  'meteomin',
#                                                                  'ones',
#                                                                  'stamp',
#                                                                  'targetlag',
#                                                                  'wh',
#                                                                  'xmas',
#                                                                  'yd',
#                                                                  ####
#                                                                  'do#wh',
#                                                                  'dado#wh',
#                                                                  'dbdo#wh',
#                                                                  'dl#nebu',
#                                                                  'targetlag#wh',
#                                                                  'meteo#yd',
#                                                                  'wh#yd',
#                                                                  ]),
#                                             },
#                            ###
#                            'InterUV'     : {
#                                             'Bsp' : tuple(['meteo',
#                                                            'meteolag',
#                                                            'meteomax',
#                                                            'meteomin',
#                                                            'ones',
#                                                            'stamp',
#                                                            'targetlag',
#                                                            'wh',
#                                                            'xmas',
#                                                            'yd',
#                                                            ####
#                                                            'do#wh',
#                                                            'dado#wh',
#                                                            'dbdo#wh',
#                                                            'dl#nebu',
#                                                            ]),
#                                             'Cuv' : tuple([
#                                                            'targetlag#wh',
#                                                            'meteo#yd',
#                                                            'wh#yd',
#                                                            ]),
#                                             },
#                            ###
#                            'sesq'        : {
#                                             'Bsp' : tuple(['meteolag',
#                                                            'meteomax',
#                                                            'meteomin',
#                                                            'ones',
#                                                            'stamp',
#                                                            'xmas',
#                                                            ####
#                                                            'do#wh',
#                                                            'dado#wh',
#                                                            'dbdo#wh',
#                                                            'dl#nebu',
#                                                            'meteo#yd',
#                                                            ]),
#                                             'Cb'  : tuple([
#                                                            'meteo',
#                                                            'targetlag',
#                                                            'wh',
#                                                            'yd',
#                                                            ]),
#                                             'Cbm' : tuple([
#                                                            'wh#yd',
#                                                            'targetlag#wh',
#                                                            ]),
#                                             },
#                            }, 
#               ############
#                'nat' : {
#                           'lowrank' :      {
#                                             'Blr' : tuple([
#                                                            'wh',
#                                                            'yd',
#                                                            'do#wh',
#                                                            'dado#wh',
#                                                            'dbdo#wh',
#                                                            'wh#yd',
#                                                            ]),
#                                             'Bsp' : tuple([
#                                                            'meteo',
#                                                            'meteolag',
#                                                            'meteomax',
#                                                            'meteomin',
#                                                            'ones',
#                                                            'stamp',
#                                                            'targetlag',
#                                                            'xmas',
#                                                            'dl#nebu',
#                                                            'targetlag#wh',
#                                                            'meteo#yd',
#                                                            ]),
#                                             },
#                           'univariate'   : {
#                                             'lbfgs_coef' : tuple([
#                                                                  'meteo',
#                                                                  'meteolag',
#                                                                  'meteomax',
#                                                                  'meteomin',
#                                                                  'ones',
#                                                                  'stamp',
#                                                                  'targetlag',
#                                                                  'wh',
#                                                                  'xmas',
#                                                                  'yd',
#                                                                  'do', 
#                                                                  'dado', 
#                                                                  'dbdo', 
#                                                                  'dl', 
#                                                                  'nebu',
#                                                                  ]),
#                                             },
#                            ###
#                            'lbfgs'       : {
#                                             'lbfgs_coef' : tuple([
#                                                                  'meteo',
#                                                                  'meteolag',
#                                                                  'meteomax',
#                                                                  'meteomin',
#                                                                  'ones',
#                                                                  'stamp',
#                                                                  'targetlag',
#                                                                  'wh',
#                                                                  'xmas',
#                                                                  'yd',
#                                                                  ####
#                                                                  'do#wh',
#                                                                  'dado#wh',
#                                                                  'dbdo#wh',
#                                                                  'dl#nebu',
#                                                                  'targetlag#wh',
#                                                                  'meteo#yd',
#                                                                  'wh#yd',
#                                                                  ]),
#                                             },
#                            ###
#                            'InterUV'     : {
#                                             'Bsp' : tuple(['meteo',
#                                                            'meteolag',
#                                                            'meteomax',
#                                                            'meteomin',
#                                                            'ones',
#                                                            'stamp',
#                                                            'targetlag',
#                                                            'wh',
#                                                            'xmas',
#                                                            'yd',
#                                                            ####
#                                                            'do#wh',
#                                                            'dado#wh',
#                                                            'dbdo#wh',
#                                                            'dl#nebu',
#                                                            ]),
#                                             'Cuv' : tuple([
#                                                            'targetlag#wh',
#                                                            'meteo#yd',
#                                                            'wh#yd',
#                                                            ]),
#                                             },
#                            ###
#                            'sesq'        : {
#                                             'Bsp' : tuple(['meteolag',
#                                                            'meteomax',
#                                                            'meteomin',
#                                                            'ones',
#                                                            'stamp',
#                                                            'xmas',
#                                                            ####
#                                                            'do#wh',
#                                                            'dado#wh',
#                                                            'dbdo#wh',
#                                                            'dl#nebu',
#                                                            'meteo#yd',
#                                                            ]),
#                                             'Cb'  : tuple([
#                                                            'meteo',
#                                                            'targetlag',
#                                                            'wh',
#                                                            'yd',
#                                                            ]),
#                                             'Cbm' : tuple([
#                                                            'wh#yd',
#                                                            'targetlag#wh',
#                                                            ]),
#                                             },
#                            }, 
#               ############
#               'full' : {
#                           'lowrank' :      {
#                                             'Blr' : tuple([
#                                                            'wh',
#                                                            'yd',
#                                                            'do#wh',
#                                                            'dado#wh',
#                                                            'dbdo#wh',
#                                                            'wh#yd',
#                                                            ]),
#                                             'Bsp' : tuple([
#                                                            'meteo',
#                                                            'meteolag',
#                                                            'meteomax',
#                                                            'meteomin',
#                                                            'ones',
#                                                            'stamp',
#                                                            'targetlag',
#                                                            'xmas',
#                                                            'dl#nebu',
#                                                            'targetlag#wh',
#                                                            'meteo#yd',
#                                                            ]),
#                                             },
#                           'univariate'   : {
#                                             'lbfgs_coef' : tuple([
#                                                                  'meteo',
#                                                                  'meteolag',
#                                                                  'meteomax',
#                                                                  'meteomin',
#                                                                  'ones',
#                                                                  'stamp',
#                                                                  'targetlag',
#                                                                  'wh',
#                                                                  'xmas',
#                                                                  'yd',
#                                                                  'do', 
#                                                                  'dado', 
#                                                                  'dbdo', 
#                                                                  'dl', 
#                                                                  'nebu',
#                                                                  ]),
#                                             },
#                            ###
#                            'lbfgs'       : {
#                                             'lbfgs_coef' : tuple([
#                                                                  'meteo',
#                                                                  'meteolag',
#                                                                  'meteomax',
#                                                                  'meteomin',
#                                                                  'ones',
#                                                                  'stamp',
#                                                                  'targetlag',
#                                                                  'wh',
#                                                                  'xmas',
#                                                                  'yd',
#                                                                  ####
#                                                                  'do#wh',
#                                                                  'dado#wh',
#                                                                  'dbdo#wh',
#                                                                  'dl#nebu',
#                                                                  'targetlag#wh',
#                                                                  'meteo#yd',
#                                                                  'wh#yd',
#                                                                  ]),
#                                             },
#                            ###
#                            'InterUV'     : {
#                                             'Bsp' : tuple(['meteo',
#                                                            'meteolag',
#                                                            'meteomax',
#                                                            'meteomin',
#                                                            'ones',
#                                                            'stamp',
#                                                            'targetlag',
#                                                            'wh',
#                                                            'xmas',
#                                                            'yd',
#                                                            ####
#                                                            'do#wh',
#                                                            'dado#wh',
#                                                            'dbdo#wh',
#                                                            'dl#nebu',
#                                                            ]),
#                                             'Cuv' : tuple([
#                                                            'targetlag#wh',
#                                                            'meteo#yd',
#                                                            'wh#yd',
#                                                            ]),
#                                             },
#                            ###
#                            'sesq'        : {
#                                             'Bsp' : tuple(['meteolag',
#                                                            'meteomax',
#                                                            'meteomin',
#                                                            'ones',
#                                                            'stamp',
#                                                            'xmas',
#                                                            ####
#                                                            'do#wh',
#                                                            'dado#wh',
#                                                            'dbdo#wh',
#                                                            'dl#nebu',
#                                                            'meteo#yd',
#                                                            ]),
#                                             'Cb'  : tuple([
#                                                            'meteo',
#                                                            'targetlag',
#                                                            'wh',
#                                                            'yd',
#                                                            ]),
#                                             'Cbm' : tuple([
#                                                            'wh#yd',
#                                                            'targetlag#wh',
#                                                            ]),
#                                             },
#                            }, 
#                }

#dikt_formula['nat'] ['unstructured'] = {'Bsp' : dikt_config['nat']['lbfgs']['lbfgs_coef']}
#dikt_formula['full']['unstructured'] = {'Bsp' : dikt_config['nat']['lbfgs']['lbfgs_coef']}
#            
#dikt_formula.update({
#                     'e2m'       : cp.deepcopy(dikt_config['nat']),
#                     'rteReg'    : cp.deepcopy(dikt_config['nat']),
#                     'admReg'    : cp.deepcopy(dikt_config['nat']),
#                     'districts' : cp.deepcopy(dikt_config['nat']),
#                     #'4Paris'    : cp.deepcopy(dikt_config['nat']),
#                     #'Normandie' : cp.deepcopy(dikt_config['nat']),
#                     })
              
############################################################################
                     
##                                Masks                                  ###

############################################################################

#active_mask =  {
#                'nebu'      : 2, # meaning 2 closest stations per post
#                'meteo'     : 2, # meaning 2 closest stations per post
#                'meteolag'  : 2, # meaning 2 closest stations per post
#                'meteosmo'  : 2, # meaning 2 closest stations per post
#                'meteodif'  : 2, # meaning 2 closest stations per post
#                'meteomin'  : 2, # meaning 2 closest stations per post
#                'meteomax'  : 2, # meaning 2 closest stations per post
#                'targetlag' : 1,
#                'wt'        : 1,
#                }   

              
###############################################################################
                     
###                            Number of intervals                          ###                

###############################################################################

#nb_itv  =  {
#            ('Eco2mix.administrative_regions',
#             None,
#             ) : {
#                     'dado'               : 'p1',
#                     'dbdo'               : 'p1',
#                     'do'                 : 'p1',
#                     'meteo'              : 16,
#                     'meteolag'           : 16,
#                     'meteomax'           : 16,
#                     'meteomin'           : 16,
#                     'ones'               : 'p1',
#                     'stamp'              : 'p1',
#                     'targetlag'          : 4,
#                     'wh'                 : 168,
#                     'xmas'               : 'p1',
#                     'yd'                 : 128, 
#                     'dl'                 : 'p1',
#                     #'h'                  : 24,
#                     #'meteodif'           : 8,
#                     #'meteodifmeteolag'   : 8,
#                     #'meteodiftargetlag'  : 8,
#                     #'meteosmo'           : 8,
#                     #'nebu'               : 2,
#                     #'sbrk'               : 0,
#                     #'target'             : 'lin',
#                     #'targetdif'          : 8,
#                     #'wd'                 : 7,
#                     ####
#                     'do#wh'         : ('p1',84),
#                     'dado#wh'       : ('p1',84), 
#                     'dbdo#wh'       : ('p1',84), 
#                     'dl#nebu'       : ('p1',2),
#                     'targetlag#wh'  : (4,84), 
#                     'meteo#yd'      : (4,32),
#                     'wh#yd'         : (168,32),
#                     },
#            'nat' : {
#                     'dado'               : 'p1',
#                     'dbdo'               : 'p1',
#                     'do'                 : 'p1',
#                     'meteo'              : 16,
#                     'meteolag'           : 16,
#                     'meteomax'           : 16,
#                     'meteomin'           : 16,
#                     'ones'               : 'p1',
#                     'stamp'              : 'p1',
#                     'targetlag'          : 4,
#                     'wh'                 : 168,
#                     'xmas'               : 'p1',
#                     'yd'                 : 128, 
#                     'dl'                 : 'p1',
#                     #'h'                  : 24,
#                     #'meteodif'           : 8,
#                     #'meteodifmeteolag'   : 8,
#                     #'meteodiftargetlag'  : 8,
#                     #'meteosmo'           : 8,
#                     #'nebu'               : 2,
#                     #'sbrk'               : 0,
#                     #'target'             : 'lin',
#                     #'targetdif'          : 8,
#                     #'wd'                 : 7,
#                     ####
#                     'do#wh'         : ('p1',84),
#                     'dado#wh'       : ('p1',84), 
#                     'dbdo#wh'       : ('p1',84), 
#                     'dl#nebu'       : ('p1',2),
#                     'targetlag#wh'  : (4,84), 
#                     'meteo#yd'      : (4,32),
#                     'wh#yd'         : (168,32),
#                     },
#            'nat_cbm' : {
#                         #'dado'               : 'p1',
#                         #'dbdo'               : 'p1',
#                         #'do'                 : 'p1',
#                         'meteo'              : 16,
#                         'meteolag'           : 16,
#                         'meteomax'           : 16,
#                         'meteomin'           : 16,
#                         'ones'               : 'p1',
#                         'stamp'              : 'p1',
#                         'targetlag'          : 'p1',
#                         'wh'                 : 168,
#                         'xmas'               : 'p1',
#                         'yd'                 : 128, 
#                         #'dl'                 : 'p1',
#                         ####
#                         'do#wh'         : ('p1',168),
#                         'dado#wh'       : ('p1',168), 
#                         'dbdo#wh'       : ('p1',168), 
#                         'dl#nebu'       : ('p1',2),
#                         'targetlag#wh'  : ('p1',168), 
#                         'meteo#yd'      : (16,128),
#                         'wh#yd'         : (168,128),
#                         },
#            'full' : {
#                     'dado'               : 'p1',
#                     'dbdo'               : 'p1',
#                     'dl'                 : 'p1',
#                     'do'                 : 'p1',
#                     'meteo'              : 8,
#                     'meteolag'           : 8,
#                     'meteomax'           : 8,
#                     'meteomin'           : 8,
#                     #'nebu'               : 8,
#                     'ones'               : 'p1',
#                     'stamp'              : 'p1',
#                     'target'             : 'p1',
#                     'targetlag'          : 2,
#                     'wh'                 : 168,
#                     'xmas'               : 'p1',
#                     'yd'                 : 16,
#                     #'h'                  : 24,
#                     #'meteodif'           : 8,
#                     #'meteodifmeteolag'   : 8,
#                     #'meteodiftargetlag'  : 8,
#                     #'meteosmo'           : 8,
#                     #'sbrk'               : 0,
#                     #'targetdif'          : 8, 
#                     #'wd'                 : 7,
#                     ####
#                     'do#wh'         : ('p1',168),
#                     'dado#wh'       : ('p1',168), 
#                     'dbdo#wh'       : ('p1',168), 
#                     'dl#nebu'       : ('p1',2),
#                     'targetlag#wh'  : ('p1',42), 
#                     'meteo#yd'      : (8,16),
#                     'wh#yd'         : (168,16),
#                     },
#            'full_cbm' : {
#                     'dado'               : 'p1',
#                     'dbdo'               : 'p1',
#                     'dl'                 : 'p1',
#                     'do'                 : 'p1',
#                     'meteo'              : 8,
#                     'meteolag'           : 8,
#                     'meteomax'           : 8,
#                     'meteomin'           : 8,
#                     #'nebu'               : 8,
#                     'ones'               : 'p1',
#                     'stamp'              : 'p1',
#                     'target'             : 'p1',
#                     'targetlag'          : 2,
#                     'wh'                 : 168,
#                     'xmas'               : 'p1',
#                     'yd'                 : 16,
#                     #'h'                  : 24,
#                     #'meteodif'           : 8,
#                     #'meteodifmeteolag'   : 8,
#                     #'meteodiftargetlag'  : 8,
#                     #'meteosmo'           : 8,
#                     #'sbrk'               : 0,
#                     #'targetdif'          : 8, 
#                     #'wd'                 : 7,
#                     ####
#                     'do#wh'         : ('p1',168),
#                     'dado#wh'       : ('p1',168), 
#                     'dbdo#wh'       : ('p1',168), 
#                     'dl#nebu'       : ('p1',2),
#                     'targetlag#wh'  : (2,168), # Both must b the same size as the univariate effects
#                     'meteo#yd'      : (8,16),
#                     'wh#yd'         : (168,16),
#                     },
#            'hrch' : {
#                     'dado'          : 'p1',
#                     'dbdo'          : 'p1',
#                     'do'            : 'p1',
#                     'meteo'         : 16,
#                     'meteolag'      : 16,
#                     'meteomax'      : 16,
#                     'meteomin'      : 16,
#                     'ones'          : 'p1',
#                     'stamp'         : 'p1',
#                     'targetlag'     : 4,
#                     'wh'            : (7, 14, 42, 84, 168),
#                     'xmas'          : 'p1',
#                     'yd'            : (8, 16, 32, 64, 128), 
#                     ####
#                     'do#wh'         : ('p1',(7, 14, 42, 84, 168)),
#                     'dado#wh'       : ('p1',(7, 14, 42, 84, 168)), 
#                     'dbdo#wh'       : ('p1',(7, 14, 42, 84, 168)), 
#                     'dl#nebu'       : ('p1',2),
#                     'targetlag#wh'  : (4,84), 
#                     'meteo#yd'      : (4,32),
#                     'wh#yd'         : ((7, 14, 42, 84, 168),4),
#                     },
#            }
            
#nb_itv.update({
#               'e2m'       : cp.deepcopy(nb_itv['nat']),
#               'rteReg'    : cp.deepcopy(nb_itv['nat']),
#               'admReg'    : cp.deepcopy(nb_itv['nat']),
#               'districts' : cp.deepcopy(nb_itv['nat']),
#               #'4Paris'    : cp.deepcopy(nb_itv['nat']),
#               #'Normandie' : cp.deepcopy(nb_itv['nat']),
#               })

              
###############################################################################
                     
###                             Regularization                              ###

###############################################################################

#pen_tmp = 'rsm'
#regularization_func = {
#                       ('Eco2mix.administrative_regions',
#                        None,
#                        ) : {
#                             'A'   : {'y'         :'rsm'}, 
#                             'Blr' : {
#                                      'wh'        : 'rsm',
#                                      'yd'        : '0r2sm',
#                                      'do#wh'     : '0r2sm',
#                                      'dado#wh'   : '0r2sm',
#                                      'dbdo#wh'   : '0r2sm',
#                                      'wh#yd'     : 'r2sm',
#                                      },
#                             'B' : {
#                                      'meteo'     : 'r2sm',
#                                      'meteolag'  : 'r2sm',
#                                      'meteomin'  : 'r2sm',
#                                      'meteomax'  : 'r2sm',
#                                      'ones'      : '',
#                                      'stamp'     : 'rsm',
#                                      'targetlag' : 'r2sm',
#                                      'wh'        : 'rsm',     
#                                      'xmas'      : '', 
#                                      'yd'        : 'r2sm',      
#                                      
#                                      'do#wh'        : 'r2sm',
#                                      'dado#wh'      : 'r2sm',
#                                      'dbdo#wh'      : 'r2sm',
#                                      'wh#yd'        : 'r2sm', 
#                                      
#                                      'targetlag#wh' : 'rsm',
#                                      'meteo#yd'     : 'r2sm',
#                                      'dl#nebu'      : 'r2sm',
#                                      },
#                             'Cb'  : {
#                                      'meteo'     : 'r2sm',
#                                      'targetlag' : 'r2sm',
#                                      'wh'        : 'rsm',     
#                                      'yd'        : 'r2sm', 
#                                      #'meteolag'  : 'rsm',
#                                      #'meteomin'  : 'rsm',
#                                      #'meteomax'  : 'rsm',     
#                                      },               
#                             'Cm'  : {
#                                      'do#wh'        : 'r2sm',
#                                      'dado#wh'      : 'r2sm',
#                                      'dbdo#wh'      : 'r2sm',
#                                      'wh#yd'        : 'r2sm', 
#                                      
#                                      'targetlag#wh' : 'rsm',
#                                      'meteo#yd'     : 'r2sm',
#                                      'dl#nebu'      : 'r2sm',
#                                      },
#                             'Cuv' : {
#                                      'do#wh'        : 'rsm',
#                                      'dado#wh'      : 'rsm',
#                                      'dbdo#wh'      : 'rsm',
#                                      'wh#yd'        : 'rsm', 
#                                      
#                                      'targetlag#wh' : 'rsm',
#                                      'meteo#yd'     : 'rsm',
#                                      'dl#nebu'      : 'rsm',
#                                      },
#                     'lbfgs_coef' : {
#                                     'meteo'     : 'r2sm', #'rsm', # 'lasso', # 'ncvxclasso'  # 'n2cvxclasso', #
#                                     'meteolag'  : 'r2sm', #'rsm', # 'lasso', # 'ncvxclasso'  # 'n2cvxclasso', # 
#                                     'meteomin'  : 'r2sm', #'rsm', # 'lasso', # 'ncvxclasso'  # 'n2cvxclasso', # 
#                                     'meteomax'  : 'r2sm', #'rsm', # 'lasso', # 'ncvxclasso'  # 'n2cvxclasso', # 
#                                     'ones'      : '',
#                                     'stamp'     : 'rsm', #'rsm', 
#                                     'targetlag' : 'r2sm', #'rsm', # 'lasso', # 'ncvxclasso'  # 'n2cvxclasso', #
#                                     'wh'        : 'rsm', #'rsm',      
#                                     'xmas'      : '', 
#                                     'yd'        : 'r2sm',
#                                     
#                                     # Interactions with indicators
#                                     'do#wh'        : 'r2sm',
#                                     'dado#wh'      : 'r2sm',
#                                     'dbdo#wh'      : 'r2sm',
#                                     'dl#nebu'      : 'r2sm',
#                                     
#                                     # 2-dimensional interactions
#                                     'targetlag#wh' : 'rsm',
#                                     'wh#yd'        : 'r2sm', 
#                                     'meteo#yd'     : 'r2sm',
#                                     },
#                     },
#            ###
#            'nat' : {
#                     'A'   : {'y'         :'rsm'}, 
#                     'Blr' : {
#                              'wh'        : 'rsm',
#                              'yd'        : '0r2sm',
#                              'do#wh'     : '0r2sm',
#                              'dado#wh'   : '0r2sm',
#                              'dbdo#wh'   : '0r2sm',
#                              'wh#yd'     : 'r2sm',
#                              },
#                     'Bsp' : {
#                              'meteo'     : 'r2sm',
#                              'meteolag'  : 'r2sm',
#                              'meteomin'  : 'r2sm',
#                              'meteomax'  : 'r2sm',
#                              'ones'      : '',
#                              'stamp'     : 'rsm',
#                              'targetlag' : 'r2sm',
#                              'wh'        : 'rsm',     
#                              'xmas'      : '', 
#                              'yd'        : 'r2sm',      
#                              
#                              'do#wh'        : 'r2sm',
#                              'dado#wh'      : 'r2sm',
#                              'dbdo#wh'      : 'r2sm',
#                              'wh#yd'        : 'r2sm', 
#                              
#                              'targetlag#wh' : 'rsm',
#                              'meteo#yd'     : 'r2sm',
#                              'dl#nebu'      : 'r2sm',
#                              },
#                     'Cb'  : {
#                              'meteo'     : 'r2sm',
#                              'targetlag' : 'r2sm',
#                              'wh'        : 'rsm',     
#                              'yd'        : 'r2sm', 
#                              #'meteolag'  : 'rsm',
#                              #'meteomin'  : 'rsm',
#                              #'meteomax'  : 'rsm',     
#                              },               
#                     'Cm'  : {
#                              'do#wh'        : 'r2sm',
#                              'dado#wh'      : 'r2sm',
#                              'dbdo#wh'      : 'r2sm',
#                              'wh#yd'        : 'r2sm', 
#                              
#                              'targetlag#wh' : 'rsm',
#                              'meteo#yd'     : 'r2sm',
#                              'dl#nebu'      : 'r2sm',
#                              },
#                     'Cuv' : {
#                              'do#wh'        : 'rsm',
#                              'dado#wh'      : 'rsm',
#                              'dbdo#wh'      : 'rsm',
#                              'wh#yd'        : 'rsm', 
#                              
#                              'targetlag#wh' : 'rsm',
#                              'meteo#yd'     : 'rsm',
#                              'dl#nebu'      : 'rsm',
#                              },
#                     'lbfgs_coef' : {
#                                     'meteo'     : 'r2sm', #'rsm', # 'lasso', # 'ncvxclasso'  # 'n2cvxclasso', #
#                                     'meteolag'  : 'r2sm', #'rsm', # 'lasso', # 'ncvxclasso'  # 'n2cvxclasso', # 
#                                     'meteomin'  : 'r2sm', #'rsm', # 'lasso', # 'ncvxclasso'  # 'n2cvxclasso', # 
#                                     'meteomax'  : 'r2sm', #'rsm', # 'lasso', # 'ncvxclasso'  # 'n2cvxclasso', # 
#                                     'ones'      : '',
#                                     'stamp'     : 'rsm', #'rsm', 
#                                     'targetlag' : 'r2sm', #'rsm', # 'lasso', # 'ncvxclasso'  # 'n2cvxclasso', #
#                                     'wh'        : 'rsm', #'rsm',      
#                                     'xmas'      : '', 
#                                     'yd'        : 'r2sm',
#                                     
#                                     # Interactions with indicators
#                                     'do#wh'        : 'r2sm',
#                                     'dado#wh'      : 'r2sm',
#                                     'dbdo#wh'      : 'r2sm',
#                                     'dl#nebu'      : 'r2sm',
#                                     
#                                     # 2-dimensional interactions
#                                     'targetlag#wh' : 'rsm',
#                                     'wh#yd'        : 'r2sm', 
#                                     'meteo#yd'     : 'r2sm',
#                                     },
#                     },
#            ###
#            'row2sm' : {
#                     'A'   : {'y'         :'rsm'}, 
#                     'Blr' : {
#                              'wh'        : 'rsm',
#                              'yd'        : '0r2sm',
#                              'do#wh'     : '0r2sm',
#                              'dado#wh'   : '0r2sm',
#                              'dbdo#wh'   : '0r2sm',
#                              'wh#yd'     : 'r2sm',
#                              },
#                     'Bsp' : {
#                              'meteo'     : 'r2sm',
#                              'meteolag'  : 'r2sm',
#                              'meteomin'  : 'r2sm',
#                              'meteomax'  : 'r2sm',
#                              'ones'      : '',
#                              'stamp'     : 'rsm',
#                              'targetlag' : 'r2sm',
#                              'wh'        : 'rsm',     
#                              'xmas'      : '', 
#                              'yd'        : 'r2sm',      
#                              
#                              'do#wh'        : 'r2sm',
#                              'dado#wh'      : 'r2sm',
#                              'dbdo#wh'      : 'r2sm',
#                              'wh#yd'        : 'r2sm', 
#                              
#                              'targetlag#wh' : 'rsm',
#                              'meteo#yd'     : 'r2sm',
#                              'dl#nebu'      : 'r2sm',
#                              },
#                     'Cb'  : {
#                              'meteo'     : 'r2sm',
#                              'targetlag' : 'r2sm',
#                              'wh'        : 'rsm',     
#                              'yd'        : 'r2sm', 
#                              #'meteolag'  : 'rsm',
#                              #'meteomin'  : 'rsm',
#                              #'meteomax'  : 'rsm',     
#                              },               
#                     'Cm'  : {
#                              'do#wh'        : 'r2sm',
#                              'dado#wh'      : 'r2sm',
#                              'dbdo#wh'      : 'r2sm',
#                              'wh#yd'        : 'r2sm', 
#                              
#                              'targetlag#wh' : 'rsm',
#                              'meteo#yd'     : 'r2sm',
#                              'dl#nebu'      : 'r2sm',
#                              },
#                     'Cuv' : {
#                              'do#wh'        : 'rsm',
#                              'dado#wh'      : 'rsm',
#                              'dbdo#wh'      : 'rsm',
#                              'wh#yd'        : 'rsm', 
#                              
#                              'targetlag#wh' : 'rsm',
#                              'meteo#yd'     : 'rsm',
#                              'dl#nebu'      : 'rsm',
#                              },
#                     'lbfgs_coef' : {
#                                     'meteo'     : 'row2sm', #'rsm', # 'lasso', # 'ncvxclasso'  # 'n2cvxclasso', #
#                                     'meteolag'  : 'row2sm', #'rsm', # 'lasso', # 'ncvxclasso'  # 'n2cvxclasso', # 
#                                     'meteomin'  : 'row2sm', #'rsm', # 'lasso', # 'ncvxclasso'  # 'n2cvxclasso', # 
#                                     'meteomax'  : 'row2sm', #'rsm', # 'lasso', # 'ncvxclasso'  # 'n2cvxclasso', # 
#                                     'ones'      : '',
#                                     'stamp'     : 'rsm', #'rsm', 
#                                     'targetlag' : 'row2sm', #'rsm', # 'lasso', # 'ncvxclasso'  # 'n2cvxclasso', #
#                                     'wh'        : 'rsm', #'rsm',      
#                                     'xmas'      : '', 
#                                     'yd'        : 'row2sm',
#                                     
#                                     'do#wh'        : 'row2sm',
#                                     'dado#wh'      : 'row2sm',
#                                     'dbdo#wh'      : 'row2sm',
#                                     'wh#yd'        : 'rsm', 
#                                     
#                                     'targetlag#wh' : 'rsm',
#                                     'meteo#yd'     : 'row2sm',
#                                     'dl#nebu'      : 'row2sm',
#                                     },
#                     },
#            'full' : {
#                     'A'   : {'y'         :'rsm'}, 
#                     'Blr' : {
#                              'wh'        : 'rsm',
#                              'yd'        : '0r2sm',
#                              'do#wh'     : '0r2sm',
#                              'dado#wh'   : '0r2sm',
#                              'dbdo#wh'   : '0r2sm',
#                              'wh#yd'     : 'r2sm',
#                              },
#                     'Bsp' : {
#                              'meteo'     : 'r2sm',
#                              'meteolag'  : 'r2sm',
#                              'meteomin'  : 'r2sm',
#                              'meteomax'  : 'r2sm',
#                              'ones'      : '',
#                              'stamp'     : 'rsm',
#                              'targetlag' : 'r2sm',
#                              'wh'        : 'rsm',     
#                              'xmas'      : '', 
#                              'yd'        : 'r2sm',      
#                              
#                              'do#wh'        : 'r2sm',
#                              'dado#wh'      : 'r2sm',
#                              'dbdo#wh'      : 'r2sm',
#                              'wh#yd'        : 'rsm', 
#                              
#                              'targetlag#wh' : 'rsm',
#                              'meteo#yd'     : 'r2sm',
#                              'dl#nebu'      : 'r2sm',
#                              },
#                     'Cb'  : {
#                              'meteo'     : 'r2sm',
#                              'targetlag' : 'r2sm',
#                              'wh'        : 'rsm',     
#                              'yd'        : 'r2sm', 
#                              #'meteolag'  : 'rsm',
#                              #'meteomin'  : 'rsm',
#                              #'meteomax'  : 'rsm',     
#                              },               
#                     'Cm'  : {
#                              'do#wh'        : 'r2sm',
#                              'dado#wh'      : 'r2sm',
#                              'dbdo#wh'      : 'r2sm',
#                              'wh#yd'        : 'r2sm', 
#                              
#                              'targetlag#wh' : 'rsm',
#                              'meteo#yd'     : 'r2sm',
#                              'dl#nebu'      : 'r2sm',
#                              },
#                     'Cuv' : {
#                              'do#wh'        : 'rsm',
#                              'dado#wh'      : 'rsm',
#                              'dbdo#wh'      : 'rsm',
#                              'wh#yd'        : 'rsm', 
#                              
#                              'targetlag#wh' : 'rsm',
#                              'meteo#yd'     : 'rsm',
#                              'dl#nebu'      : 'rsm',
#                              },
#                     'lbfgs_coef' : {
#                                     'meteo'     : 'r2sm', #'rsm', # 'lasso', # 'ncvxclasso'  # 'n2cvxclasso', #
#                                     'meteolag'  : 'r2sm', #'rsm', # 'lasso', # 'ncvxclasso'  # 'n2cvxclasso', # 
#                                     'meteomin'  : 'r2sm', #'rsm', # 'lasso', # 'ncvxclasso'  # 'n2cvxclasso', # 
#                                     'meteomax'  : 'r2sm', #'rsm', # 'lasso', # 'ncvxclasso'  # 'n2cvxclasso', # 
#                                     'ones'      : '',
#                                     'stamp'     : 'rsm', #'rsm', 
#                                     'targetlag' : 'r2sm', #'rsm', # 'lasso', # 'ncvxclasso'  # 'n2cvxclasso', #
#                                     'wh'        : 'rsm', #'rsm',      
#                                     'xmas'      : '', 
#                                     'yd'        : 'r2sm',
#                                     
#                                     'do#wh'        : 'r2sm',
#                                     'dado#wh'      : 'r2sm',
#                                     'dbdo#wh'      : 'r2sm',
#                                     'wh#yd'        : 'rsm', 
#                                     
#                                     'targetlag#wh' : 'rsm',
#                                     'meteo#yd'     : 'r2sm',
#                                     'dl#nebu'      : 'r2sm',
#                                     },
#                     },
#            'hrch' : {
#                      'Blr' : {
#                              'wh'           : 'rlasso',
#                              'yd'           : 'rlasso',
#                              ####
#                              'do#wh'        : 'rlasso',
#                              'dado#wh'      : 'rlasso',
#                              'dbdo#wh'      : 'rlasso',
#                              'wh#yd'        : 'rlasso',
#                              },
#                     'Bsp' : {
#                              'meteo'     : 'r2sm',
#                              'meteolag'  : 'r2sm',
#                              'meteomin'  : 'r2sm',
#                              'meteomax'  : 'r2sm',
#                              'ones'      : '',
#                              'stamp'     : 'rsm',
#                              'targetlag' : 'r2sm',
#                              'xmas'      : '', 
#                              ####
#                              'targetlag#wh' : 'r2sm',
#                              'meteo#yd'     : 'r2sm',
#                              'dl#nebu'      : 'r2sm',
#                              }
#                      },
#             }

                       
#pen_base.update({
#                 'e2m'       : cp.deepcopy(pen_base['nat']),
#                 'rteReg'    : cp.deepcopy(pen_base['nat']),
#                 'admReg'    : cp.deepcopy(pen_base['nat']),
#                 'districts' : cp.deepcopy(pen_base['nat']),
#                 })
              
###############################################################################
                     
###                             Alphas                                      ###

###############################################################################

#alpha_Blr       = 1e-6
#alpha_Cb        = 1e-8
#factor_nb_posts = 4e2
##alpha_lbfgs_uni = 1e-8
#regularization_coef = {
#                       ('Eco2mix.administrative_regions',
#                        None,
#                        ): {
#                               'A'     : {'y' : 1e-2},
#                               'Cuv'   : {     
#                                        'do#wh'        : 1e-8,
#                                        'dado#wh'      : 1e-8, 
#                                        'dbdo#wh'      : 1e-8, 
#                                        'dl#nebu'      : 1e-8,
#                                        'targetlag#wh' : 1e-6,
#                                        'meteo#yd'     : 1e-8,
#                                        'wh#yd'        : 1e-8,
#                                        }, 
#                               'B' : {     
#                                            'yd'        : 1e-3,
#                                            'wh'        : 1e-8,
#                                            'meteo'     : 1e-4,
#                                            'meteolag'  : 1e-2,
#                                            'meteomin'  : 1e-1,
#                                            'meteomax'  : 1e-2,
#                                            'stamp'     : 1e-3,
#                                            'targetlag' : 1e-4,
#                                            ####
#                                            'do#wh'        : 1e-8,
#                                            'dado#wh'      : 1e-8, 
#                                            'dbdo#wh'      : 1e-8, 
#                                            'dl#nebu'      : 1e-8,
#                                            'targetlag#wh' : 1e-6,
#                                            'meteo#yd'     : 1e-4,
#                                            'wh#yd'        : 1e-5,
#                                            }, 
#                             },
#                       'nat': {
#                               'A'     : {'y' : 1e-2},
#                               'Cuv'   : {     
#                                        'do#wh'        : 1e-8,
#                                        'dado#wh'      : 1e-8, 
#                                        'dbdo#wh'      : 1e-8, 
#                                        'dl#nebu'      : 1e-8,
#                                        'targetlag#wh' : 1e-6,
#                                        'meteo#yd'     : 1e-8,
#                                        'wh#yd'        : 1e-8,
#                                        }, 
#                               'lbfgs_coef' : {     
#                                            'yd'        : 1e-3,
#                                            'wh'        : 1e-8,
#                                            'meteo'     : 1e-4,
#                                            'meteolag'  : 1e-2,
#                                            'meteomin'  : 1e-1,
#                                            'meteomax'  : 1e-2,
#                                            'stamp'     : 1e-3,
#                                            'targetlag' : 1e-4,
#                                            ####
#                                            'do#wh'        : 1e-8,
#                                            'dado#wh'      : 1e-8, 
#                                            'dbdo#wh'      : 1e-8, 
#                                            'dl#nebu'      : 1e-8,
#                                            'targetlag#wh' : 1e-6,
#                                            'meteo#yd'     : 1e-4,
#                                            'wh#yd'        : 1e-5,
#                                            }, 
#                             }, 
#                       'row2sm': {
#                               'A'     : {'y' : 1e-2},
#                               'Cuv'   : {     
#                                        'do#wh'        : 1e-8,
#                                        'dado#wh'      : 1e-8, 
#                                        'dbdo#wh'      : 1e-8, 
#                                        'dl#nebu'      : 1e-8,
#                                        'targetlag#wh' : 1e-6,
#                                        'meteo#yd'     : 1e-8,
#                                        'wh#yd'        : 1e-8,
#                                        }, 
#                               'lbfgs_coef' : {     
#                                            'yd'           : (1e-4*factor_nb_posts, 1.1),
#                                            'wh'           : 1e-8,
#                                            'meteo'        : (1e-4*factor_nb_posts, 1.1),
#                                            'meteolag'     : (1e-4*factor_nb_posts, 1.1),
#                                            'meteomin'     : (1e-4*factor_nb_posts, 1.1),
#                                            'meteomax'     : (1e-4*factor_nb_posts, 1.1),
#                                            'stamp'        : 1e-3,
#                                            'targetlag'    : (1e-4*factor_nb_posts, 1.1),
#                                            ####
#                                            'do#wh'        : (1e-9*factor_nb_posts, 1.1),#1e-8, #(1e-8*factor_nb_posts, 1.1), #
#                                            'dado#wh'      : (1e-9*factor_nb_posts, 1.1),#1e-8, #(1e-8*factor_nb_posts, 1.1), #
#                                            'dbdo#wh'      : (1e-9*factor_nb_posts, 1.1),#1e-8, #(1e-8*factor_nb_posts, 1.1), #
#                                            'dl#nebu'      : (1e-9*factor_nb_posts, 1.1),#1e-8, #(1e-8*factor_nb_posts, 1.1), #
#                                            'meteo#yd'     : (1e-7*factor_nb_posts, 1.1),#1e-8, #(1e-8*factor_nb_posts, 1.1), #
#                                            ####
#                                            'targetlag#wh' : 1e-6,
#                                            'wh#yd'        : 1e-5,
#                                            }, 
#                             }, 
#                        'full' : {
#                             'A'     : {'y' : 1e-2}, 
#                             'Cuv'   : {     
#                                        'do#wh'        : 1e-8,
#                                        'dado#wh'      : 1e-8, 
#                                        'dbdo#wh'      : 1e-8, 
#                                        'dl#nebu'      : 1e-8,
#                                        'targetlag#wh' : 1e-6,
#                                        'meteo#yd'     : 1e-8,
#                                        'wh#yd'        : 1e-8,
#                                        }, 
#                             'lbfgs_coef' : {
#                                             'yd'        : 1e2,
#                                             'wh'        : 1e-5,
#                                             'meteo'     : 1e-3,
#                                             'meteolag'  : 1e-5,
#                                             'meteomin'  : 1e-2,
#                                             'meteomax'  : 1e-3,
#                                             'targetlag' : 1e-4,
#                                             'stamp'     : 1e-2,
#                                             ####
#                                             'do#wh'        : 1e-6,
#                                             'dado#wh'      : 1e-6, 
#                                             'dbdo#wh'      : 1e-6, 
#                                             'dl#nebu'      : 1e-6,
#                                             'targetlag#wh' : 1e-4,
#                                             'meteo#yd'     : 1e2,
#                                             'wh#yd'        : 1e-3,
#                                             },                           
#                               },
#                       'hrch': {
#                               'Blr' : {
#                                        'wh' : 1e-7,
#                                        'yd' : 1e-7,
#                                        ####
#                                        'do#wh'   : 1e-8,
#                                        'dado#wh' : 1e-8,
#                                        'dbdo#wh' : 1e-8,
#                                        'wh#yd'   : 1e-8,
#                                        },    
#                               'Bsp' : {      
#                                        'meteo'     : 1e-8,
#                                        'meteolag'  : 1e-2,
#                                        'meteomin'  : 1e-1,
#                                        'meteomax'  : 1e-2,
#                                        'stamp'     : 1,
#                                        'targetlag' : 1e-3,
#                                        ####
#                                        'dl#nebu'      : 1e-8,
#                                        'targetlag#wh' : 1e-6,
#                                        'meteo#yd'     : 1e-4,
#                                        },   
#                             },
#                        }
                       
#alphas_base['nat'].update({
#                           'Blr' : alphas_base['nat']['lbfgs_coef'], 
#                           'Bsp' : alphas_base['nat']['lbfgs_coef'], 
#                           'Cb'  : {k:v for k,v in alphas_base['full']['lbfgs_coef'].items() if '#' not in k}, 
#                           'Cm'  : {k:v for k,v in alphas_base['nat' ]['lbfgs_coef'].items() if '#'     in k}, 
#                           })
#alphas_base['full'].update({
#                           'Blr' : alphas_base['full']['lbfgs_coef'], 
#                           'Bsp' : alphas_base['full']['lbfgs_coef'], 
#                           'Cb'  : {k:v for k,v in alphas_base['full']['lbfgs_coef'].items() if '#' not in k}, 
#                           'Cm'  : {k:v for k,v in alphas_base['full']['lbfgs_coef'].items() if '#'     in k}, 
#                           })
#                       
#alphas_base.update({
#                    'e2m'       : cp.deepcopy(alphas_base['nat']),
#                    'rteReg'    : cp.deepcopy(alphas_base['nat']),
#                    'admReg'    : cp.deepcopy(alphas_base['nat']),
#                    'districts' : cp.deepcopy(alphas_base['nat']),
#                    })
              
################################################################################
#                     
####                             Lag                                         ###
#
################################################################################
#                     
#               
#dikt_lags = {
#             24 : {
#                   'targetlag'         : ((24,),),
#                   'meteolag'          : ((24,),),
#                   'meteomin'          : ((24,),), 
#                   'meteomax'          : ((24,),),  
#                   }, 
#             48 : {
#                   'targetlag'         : ((24,),(48,),),
#                   'meteolag'          : ((24,),(48,),),
#                   'meteomin'          : ((24,),(48,),), 
#                   'meteomax'          : ((24,),(48,),),  
#                   },
#             168 : {
#                    'targetlag'         : ((24,),(48,),(168,),),
#                    'meteolag'          : ((24,),(48,),(168,),),
#                    'meteomin'          : ((24,),(48,),(168,),), 
#                    'meteomax'          : ((24,),(48,),(168,),),  
#                    }, 
#             'gam_nat' : {
#                    'targetlag'         : ((24,),),
#                    'meteolag'          : ((24,),(48,),),
#                    'meteomin'          : ((24,),), 
#                    'meteomax'          : ((24,),),  
#                    }, 
#             'gam_full' : {
#                    'targetlag'         : ((24,),),
#                    'meteolag'          : ((24,),(48,),), 
#                    'meteomin'          : ((24,),(48,),), 
#                    'meteomax'          : ((24,),(48,),), 
#                    }, 
#             }
              
             
             
              
###############################################################################
                     
###                             Smooth                                      ###

###############################################################################
                     
               
#dikt_smo  = { 
#             'gam_full' : {
#                           'meteosmo'          : (0.99,),
#                           }, 
#             }
              
             
        
                     
                     
                     
                     
                     
           