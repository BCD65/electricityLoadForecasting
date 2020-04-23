
import pandas as pd
import copy as cp

"""
Default hyperparameters for additive features model
"""


             
###############################################################################
###                       Stopping Criteria                                 ###
###############################################################################
              
stopping_criteria = {
                     ('eCO2mix.administrative_regions',
                      None,
                      ) : {
                           'afm.algorithm.lbfgs.maxfun'  : 1e8,
                           'afm.algorithm.lbfgs.maxiter' : 1e8,                             
                           'afm.algorithm.lbfgs.pgtol'   : 1e-6,
                           'afm.algorithm.lbfgs.tol'     : 1e-12,
                           },
                     ('eCO2mix.France',
                      None,
                      ) : {
                           'afm.algorithm.lbfgs.maxfun'  : 1e8,
                           'afm.algorithm.lbfgs.maxiter' : 1e8,                             
                           'afm.algorithm.lbfgs.tol'     : 1e-12,
                           'afm.algorithm.lbfgs.pgtol'   : 1e-4,
                           },
                    ('RTE.substations',
                     None,
                     ) : {
                          'afm.algorithm.lbfgs.maxfun'  : 1e8,
                          'afm.algorithm.lbfgs.maxiter' : 1e8,                             
                          'afm.algorithm.lbfgs.tol'     : 1e-9,
                          'afm.algorithm.lbfgs.pgtol'   : 1e-5,
                          },
                     }

stopping_criteria.update({
                          ('RTE.substations','Sum')                    : cp.deepcopy(stopping_criteria['eCO2mix.France',None]),
                          ('RTE.substations','Administrative_regions') : cp.deepcopy(stopping_criteria['eCO2mix.France',None]),
                          ('RTE.substations','Regions')                : cp.deepcopy(stopping_criteria['eCO2mix.France',None]),
                          ('RTE.substations','Districts')              : cp.deepcopy(stopping_criteria['eCO2mix.France',None]),
                          })
              
###############################################################################
###                             dikt formula                                ###
###############################################################################                        

dikt_formula = {
                ### National level
                ('eCO2mix.France',
                 None,
                 ) : {
#                      'tmp'   : pd.DataFrame([
#                                              # Univariate features
#                                              ('B', ('target',        'lag',        pd.DateOffset(hours = 24)),   4,       'r2sm', 1e-4, None),
#                                              ('B', ('temperature',   '',           ''),                          16,      'r2sm', 1e-4, None),
#                                              ('B', ('temperature',   'difference', pd.DateOffset(hours = 24)),   16,      'r2sm', 1e-2, None),
#                                              ('B', ('temperature',   'maximum',    pd.DateOffset(hours = 24)),   16,      'r2sm', 1e-2, None),
#                                              ('B', ('temperature',   'minimum',    pd.DateOffset(hours = 24)),   16,      'r2sm', 1e-1, None),
#                                              ('B', ('temperature',   'smoothing',  0.99),                        16,      'r2sm', 1e-1, None),
#                                              ('B', ('timestamp',     '',           ''),                          'p1',    'rsm',  1e-3, None),
#                                              ('B', ('week_hour',     '',           ''),                          168,     'rsm',  1e-8, None),
#                                              # Bivariate features
#                                              ('B', (('target',        'lag',        pd.DateOffset(hours = 24)),
#                                                     ('week_hour',     '',           '')),                        (4,168), 'rsm',  1e-8, None),
#                                              #
#                                              ],
#                                             columns = ['coefficient',
#                                                        'input',
#                                                        'nb_intervals',
#                                                        'regularization_func',
#                                                        'regularization_coef',
#                                                        'structure',
#                                                        ],
#                                             ).set_index(['coefficient', 'input']),
                      ### National univariate
                      'univariate': pd.DataFrame([
                                                  # Univariate features
                                                  ('B', ('daylight',      '',           ''),                           'p1',    'rsm',  1e-8, None),
                                                  ('B', ('holidays',      '',           ''),                           'p1',    'rsm',  1e-8, None),
                                                  ('B', ('holidays',      'lag',        pd.DateOffset(hours = 24)),    'p1',    'rsm',  1e-8, None),
                                                  ('B', ('holidays',      'lag',      - pd.DateOffset(hours = 24)),    'p1',    'rsm',  1e-8, None),
                                                  ('B', ('nebulosity',    '',           ''),                           'p1',    'rsm',  1e-8, None),
                                                  ('B', ('target',        'lag',        pd.DateOffset(hours = 24)),     4,      'r2sm', 1e-4, None),
                                                  ('B', ('temperature',   '',           ''),                            16,     'r2sm', 1e-4, None),
                                                  ('B', ('temperature',   'lag',        pd.DateOffset(hours = 24)),     16,     'r2sm', 1e-2, None),
                                                  ('B', ('temperature',   'maximum',    pd.DateOffset(hours = 24)),     16,     'r2sm', 1e-2, None),
                                                  ('B', ('temperature',   'minimum',    pd.DateOffset(hours = 24)),     16,     'r2sm', 1e-1, None),
                                                  ('B', ('timestamp',     '',           ''),                           'p1',    'rsm',  1e-3, None),
                                                  ('B', ('week_hour',     '',           ''),                            168,    'rsm',  1e-8, None),
                                                  ('B', ('xmas',          '',           ''),                           'p1',    '',     1e-8, None),
                                                  ('B', ('year_day',      '',           ''),                            128,    'r2sm',  1e-8, None),
                                              ],
                                             columns = ['coefficient',
                                                        'input',
                                                        'nb_intervals',
                                                        'regularization_func',
                                                        'regularization_coef',
                                                        'structure',
                                                        ],
                                             ).set_index(['coefficient', 'input']),
                      ### National standard bivariate
                      'bivariate': pd.DataFrame([
                                                  # Univariate features
                                                  ('B', ('target',        'lag',        pd.DateOffset(hours = 24)),     4,       'r2sm', 1e-4, None),
                                                  ('B', ('temperature',   '',           ''),                            16,      'r2sm', 1e-4, None),
                                                  ('B', ('temperature',   'lag',        pd.DateOffset(hours = 24)),     16,      'r2sm', 1e-2, None),
                                                  ('B', ('temperature',   'maximum',    pd.DateOffset(hours = 24)),     16,      'r2sm', 1e-2, None),
                                                  ('B', ('temperature',   'minimum',    pd.DateOffset(hours = 24)),     16,      'r2sm', 1e-1, None),
                                                  ('B', ('timestamp',     '',           ''),                            'p1',    'rsm',  1e-3, None),
                                                  ('B', ('week_hour',     '',           ''),                            168,     'rsm',  1e-8, None),
                                                  ('B', ('xmas',          '',           ''),                            'p1',    '',     1e-8, None),
                                                  ('B', ('year_day',      '',           ''),                            128,     'r2sm',  1e-8, None),
                                                  # Bivariate features
                                                  ('B', (('holidays',     '',           ''),
                                                         ('week_hour',    '',           '')),                          ('p1',84), 'r2sm',  1e-8, None),
                                                  ('B', (('holidays',     '',           pd.DateOffset(hours = 24)),
                                                         ('week_hour',    '',           '')),                          ('p1',84), 'r2sm',  1e-8, None),
                                                  ('B', (('holidays',     '',         - pd.DateOffset(hours = 24)),
                                                         ('week_hour',    '',           '')),                          ('p1',84), 'r2sm',  1e-8, None),
                                                  ('B', (('daylight',     'lag',        pd.DateOffset(hours = 24)),
                                                         ('nebulosity',   '',           '')),                          ('p1',2),  'r2sm',  1e-8, None),
                                                  ('B', (('target',       'lag',        pd.DateOffset(hours = 24)),
                                                         ('week_hour',    '',           '')),                          (4,84),    'rsm',  1e-8, None),
                                                  ('B', (('temperature',  '',           ''),
                                                         ('year_day',     '',           '')),                          (4,32),    'r2sm',  1e-8, None),
                                                  ('B', (('week_hour',    'lag',        pd.DateOffset(hours = 24)),
                                                         ('year_day',     '',           '')),                          (168,32),  'r2sm',  1e-8, None),
                                             ],
                                             columns = ['coefficient',
                                                        'input',
                                                        'nb_intervals',
                                                        'regularization_func',
                                                        'regularization_coef',
                                                        'structure',
                                                        ],
                                             ).set_index(['coefficient', 'input']),
#                      ### Bivariate with low-rank constraints on the bivariate coefficients
#                      'interUV': pd.DataFrame([
#                                                  # Univariate features
#                                                  ('B', ('target',        'lag',        pd.DateOffset(hours = 24)),     4,       'r2sm', 1e-4, None),
#                                                  ('B', ('temperature',   '',           ''),                            16,      'r2sm', 1e-4, None),
#                                                  ('B', ('temperature',   'lag',        pd.DateOffset(hours = 24)),     16,      'r2sm', 1e-2, None),
#                                                  ('B', ('temperature',   'maximum',    pd.DateOffset(hours = 24)),     16,      'r2sm', 1e-2, None),
#                                                  ('B', ('temperature',   'minimum',    pd.DateOffset(hours = 24)),     16,      'r2sm', 1e-1, None),
#                                                  ('B', ('timestamp',     '',           ''),                            'p1',    'rsm',  1e-3, None),
#                                                  ('B', ('week_hour',     '',           ''),                            168,     'rsm',  1e-8, None),
#                                                  ('B', ('xmas',          '',           ''),                            'p1',    'rsm',  1e-8, None),
#                                                  ('B', ('year_day',      '',           ''),                            168,     'rsm',  1e-8, None),
#                                                  # Bivariate features
#                                                  ('Cuv', (('target',       'lag',        pd.DateOffset(hours = 24)),
#                                                           ('week_hour',    '',           '')),                          (4,168),    'rsm',  1e-8, None),
#                                                  ('Cuv', (('temperature',  '',           ''),
#                                                           ('year_day',     '',           '')),                          (4,168),    'rsm',  1e-8, None),
#                                                  ('Cuv', (('week_hour',    'lag',        pd.DateOffset(hours = 24)),
#                                                           ('year_day',     '',           '')),                          (4,168),    'rsm',  1e-8, None),
#                                             ],
#                                             columns = ['coefficient',
#                                                        'input',
#                                                        'nb_intervals',
#                                                        'regularization_func',
#                                                        'regularization_coef',
#                                                        'structure',
#                                                        ],
#                                             ).set_index(['coefficient', 'input']),
                      ### Sesquivariate model
                      'sesq': pd.DataFrame([
                                                  # Univariate independent features
                                                  ('B', ('temperature',   '',           ''),                            16,      'r2sm', 1e-4, None),
                                                  ('B', ('temperature',   'lag',        pd.DateOffset(hours = 24)),     16,      'r2sm', 1e-2, None),
                                                  ('B', ('temperature',   'maximum',    pd.DateOffset(hours = 24)),     16,      'r2sm', 1e-2, None),
                                                  ('B', ('temperature',   'minimum',    pd.DateOffset(hours = 24)),     16,      'r2sm', 1e-1, None),
                                                  ('B', ('timestamp',     '',           ''),                            'p1',    'rsm',  1e-3, None),
                                                  ('B', ('xmas',          '',           ''),                            'p1',    '',     1e-8, None),
                                                  # Bivariate independent features
                                                  ('B', (('holidays',     '',           ''),
                                                         ('week_hour',    '',           '')),                          ('p1',84),    'r2sm',  1e-8, None),
                                                  ('B', (('holidays',     '',           pd.DateOffset(hours = 24)),
                                                         ('week_hour',    '',           '')),                          ('p1',84),    'r2sm',  1e-8, None),
                                                  ('B', (('holidays',     '',         - pd.DateOffset(hours = 24)),
                                                         ('week_hour',    '',           '')),                          ('p1',84),    'r2sm',  1e-8, None),
                                                  ('B', (('daylight',     'lag',        pd.DateOffset(hours = 24)),
                                                         ('nebulosity',   '',           '')),                          ('p1',2),     'r2sm',  1e-8, None),
                                                  ('B', (('temperature',  '',           ''),
                                                         ('year_day',     '',           '')),                          (16,128),     'r2sm',  1e-8, None),
                                                  # Univariate linked features
                                                  ('Cb', ('target',        'lag',        pd.DateOffset(hours = 24)),     'p1',       'r2sm', 1e-4, None),
                                                  ('Cb', ('week_hour',     '',           ''),                            168,        'rsm',  1e-8, None),
                                                  ('Cb', ('year_day',      '',           ''),                            128,        'r2sm', 1e-8, None),
                                                  # Bivariate linked features
                                                  ('Cbm', (('target',       'lag',        pd.DateOffset(hours = 24)),
                                                           ('week_hour',    '',           '')),                          ('p1',168), 'rsm',  1e-8, None),
                                                  ('Cbm', (('week_hour',    'lag',        pd.DateOffset(hours = 24)),
                                                           ('year_day',     '',           '')),                          (168,128),  'r2sm',  1e-8, None),
                                             ],
                                             columns = ['coefficient',
                                                        'input',
                                                        'nb_intervals',
                                                        'regularization_func',
                                                        'regularization_coef',
                                                        'structure',
                                                        ],
                                             ).set_index(['coefficient', 'input']),
                      },



#                ### Regions level
#                ('eCO2mix.administrative_regions',
#                 None,
#                 ) : {
#                      'tmp'   : pd.DataFrame([
#                                              # Univariate features
#                                              ('B', ('target',        'lag',        pd.DateOffset(hours = 24)),   4,       'r2sm', 1e-4),
#                                              ('B', ('temperature',   '',           ''),                          16,      'r2sm', 1e-4),
#                                              ('B', ('temperature',   'difference', pd.DateOffset(hours = 24)),   16,      'r2sm', 1e-2),
#                                              ('B', ('temperature',   'maximum',    pd.DateOffset(hours = 24)),   16,      'r2sm', 1e-2),
#                                              ('B', ('temperature',   'minimum',    pd.DateOffset(hours = 24)),   16,      'r2sm', 1e-1),
#                                              ('B', ('temperature',   'smoothing',  0.99),                        16,      'r2sm', 1e-1),
#                                              ('B', ('timestamp',     '',           ''),                          'p1',    'rsm',  1e-3),
#                                              ('B', ('week_hour',     '',           ''),                          168,     'rsm',  1e-8),
#                                              # Bivariate features
#                                              ('B', (('target',        'lag',       pd.DateOffset(hours = 24)),
#                                                     ('week_hour',     '',          '')),                        (4,168),  'rsm',  1e-8),
#                                              #
#                                              ],
#                                             columns = ['coefficient',
#                                                        'input',
#                                                        'nb_intervals',
#                                                        'regularization_func',
#                                                        'regularization_coef',
#                                                        ],
#                                             ).set_index(['coefficient', 'input'])
#                      },
                }



                            # Choose which variables should be part of the low-rank components and ot the unstructured components
                            # Blr        for low-rank (over the substations) formulation with a first-order descent algorithm
                            # Bsp        for independent models and unstructured coedfficients with a first-order descent algorithm
                            # lbfgs_coef for independent models and unstructured coedfficients with a first-order descent algorithm
                            # Cuv        for independent models and low-rank interactions
                            # Cb         for independent models and the univariate part of the sesquivariate constraint
                            # Cbm        for independent models and the sesquivariate constraint for the interactions
                                                                                        

####################################################################################
                                  

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


              
###############################################################################
                     
###                            Number of intervals                          ###                

###############################################################################

#nb_itv  =  {
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

#regularization_func = {
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

#factor_nb_posts = 4e2
                            
#regularization_coef = {
#                       'nat': {
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
                 
