
"""
Default hyperparameters for additive features model
"""

import pandas as pd
import copy   as cp

###############################################################################
###                             dikt formula                                ###
###############################################################################                        
factor_nb_posts = 4e2

dikt_formula = {
                ### National level
                ('eCO2mix.France',
                 None,
                 ) : {
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
                                                  ('B', ('year_day',      '',           ''),                            128,    'r2sm', 1e-3, None),
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
                                                  ('B', ('year_day',      '',           ''),                            128,     'r2sm', 1e-3, None),
                                                  # Bivariate features
                                                  ('B', (('holidays',     '',           ''),
                                                         ('week_hour',    '',           '')),                          ('p1',84), 'r2sm',  1e-8, None),
                                                  ('B', (('holidays',     'lag',        pd.DateOffset(hours = 24)),
                                                         ('week_hour',    '',           '')),                          ('p1',84), 'r2sm',  1e-8, None),
                                                  ('B', (('holidays',     'lag',      - pd.DateOffset(hours = 24)),
                                                         ('week_hour',    '',           '')),                          ('p1',84), 'r2sm',  1e-8, None),
                                                  ('B', (('daylight',     '',           ''),
                                                         ('nebulosity',   '',           '')),                          ('p1',2),  'r2sm',  1e-8, None),
                                                  ('B', (('target',       'lag',        pd.DateOffset(hours = 24)),
                                                         ('week_hour',    '',           '')),                          (4,84),    'rsm',   1e-6, None),
                                                  ('B', (('temperature',  '',           ''),
                                                         ('year_day',     '',           '')),                          (4,32),    'r2sm',  1e-4, None),
                                                  ('B', (('week_hour',    '',           ''),
                                                         ('year_day',     '',           '')),                          (168,32),  'r2sm',  1e-5, None),
                                             ],
                                             columns = ['coefficient',
                                                        'input',
                                                        'nb_intervals',
                                                        'regularization_func',
                                                        'regularization_coef',
                                                        'structure',
                                                        ],
                                             ).set_index(['coefficient', 'input']),
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
                                                  ('B', (('holidays',     'lag',        pd.DateOffset(hours = 24)),
                                                         ('week_hour',    '',           '')),                          ('p1',84),    'r2sm',  1e-8, None),
                                                  ('B', (('holidays',     'lag',      - pd.DateOffset(hours = 24)),
                                                         ('week_hour',    '',           '')),                          ('p1',84),    'r2sm',  1e-8, None),
                                                  ('B', (('daylight',     '',           ''),
                                                         ('nebulosity',   '',           '')),                          ('p1',2),     'r2sm',  1e-8, None),
                                                  ('B', (('temperature',  '',           ''),
                                                         ('year_day',     '',           '')),                          (16,128),     'r2sm',  1e-4, None),
                                                  # Univariate linked features
                                                  ('Cb', ('target',        'lag',        pd.DateOffset(hours = 24)),     'p1',       'r2sm',  1e-4, None),
                                                  ('Cb', ('week_hour',     '',           ''),                            168,        'rsm',   1e-8, None),
                                                  ('Cb', ('year_day',      '',           ''),                            128,        'r2sm',  1e-3, None),
                                                  # Bivariate linked features
                                                  ('Cbm', (('target',       'lag',       pd.DateOffset(hours = 24)),
                                                           ('week_hour',    '',          '')),                          ('p1',168),  'rsm',   1e-6, None),
                                                  ('Cbm', (('week_hour',    '',          ''),
                                                           ('year_day',     '',          '')),                          (168,128),   'r2sm',  1e-5, None),
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
                ### National level
                ('RTE.substations',
                 None,
                 ) : {
                      ### Local univariate
                      'univariate': pd.DataFrame([
                                                  # Univariate features
                                                  ('B', ('daylight',      '',           ''),                           'p1',    'rsm',  1e-8, None),
                                                  ('B', ('holidays',      '',           ''),                           'p1',    'rsm',  1e-8, None),
                                                  ('B', ('holidays',      'lag',        pd.DateOffset(hours = 24)),    'p1',    'rsm',  1e-8, None),
                                                  ('B', ('holidays',      'lag',      - pd.DateOffset(hours = 24)),    'p1',    'rsm',  1e-8, None),
                                                  ('B', ('nebulosity',    '',           ''),                           'p1',    'rsm',  1e-8, None),
                                                  ('B', ('target',        'lag',        pd.DateOffset(hours = 24)),     2,      'r2sm', 1e-4, None),
                                                  ('B', ('temperature',   '',           ''),                            8,      'r2sm', 1e-3, None),
                                                  ('B', ('temperature',   'lag',        pd.DateOffset(hours = 24)),     8,      'r2sm', 1e-5, None),
                                                  ('B', ('temperature',   'maximum',    pd.DateOffset(hours = 24)),     8,      'r2sm', 1e-3, None),
                                                  ('B', ('temperature',   'minimum',    pd.DateOffset(hours = 24)),     8,      'r2sm', 1e-2, None),
                                                  ('B', ('timestamp',     '',           ''),                           'p1',    'rsm',  1e-2, None),
                                                  ('B', ('week_hour',     '',           ''),                            168,    'rsm',  1e-5, None),
                                                  ('B', ('xmas',          '',           ''),                           'p1',    '',     1e-8, None),
                                                  ('B', ('year_day',      '',           ''),                            16,     'r2sm', 1e2,  None),
                                              ],
                                             columns = ['coefficient',
                                                        'input',
                                                        'nb_intervals',
                                                        'regularization_func',
                                                        'regularization_coef',
                                                        'structure',
                                                        ],
                                             ).set_index(['coefficient', 'input']),
                      ### Local standard bivariate
                      'bivariate': pd.DataFrame([
                                                  # Univariate features
                                                  ('B', ('target',        'lag',        pd.DateOffset(hours = 24)),     2,       'r2sm', 1e-4, None),
                                                  ('B', ('temperature',   '',           ''),                            8,       'r2sm', 1e-3, None),
                                                  ('B', ('temperature',   'lag',        pd.DateOffset(hours = 24)),     8,       'r2sm', 1e-5, None),
                                                  ('B', ('temperature',   'maximum',    pd.DateOffset(hours = 24)),     8,       'r2sm', 1e-3, None),
                                                  ('B', ('temperature',   'minimum',    pd.DateOffset(hours = 24)),     8,       'r2sm', 1e-2, None),
                                                  ('B', ('timestamp',     '',           ''),                            'p1',    'rsm',  1e-2, None),
                                                  ('B', ('week_hour',     '',           ''),                            168,     'rsm',  1e-5, None),
                                                  ('B', ('xmas',          '',           ''),                            'p1',    '',     1e-8, None),
                                                  ('B', ('year_day',      '',           ''),                            16,      'r2sm', 1e2,  None),
                                                  # Bivariate features
                                                  ('B', (('daylight',     '',           ''),
                                                         ('nebulosity',   '',           '')),                          ('p1',2),   'r2sm',  1e-6, None),
                                                  ('B', (('holidays',     '',           ''),
                                                         ('week_hour',    '',           '')),                          ('p1',168), 'r2sm',  1e-6, None),
                                                  ('B', (('holidays',     'lag',        pd.DateOffset(hours = 24)),
                                                         ('week_hour',    '',           '')),                          ('p1',168), 'r2sm',  1e-6, None),
                                                  ('B', (('holidays',     'lag',      - pd.DateOffset(hours = 24)),
                                                         ('week_hour',    '',           '')),                          ('p1',168), 'r2sm',  1e-6, None),
                                                  ('B', (('target',       'lag',        pd.DateOffset(hours = 24)),
                                                         ('week_hour',    '',           '')),                          ('p1',42),  'rsm',   1e-4, None),
                                                  ('B', (('temperature',  '',           ''),
                                                         ('year_day',     '',           '')),                          (8,16),     'r2sm',  1e2,  None),
                                                  ('B', (('week_hour',    '',           ''),
                                                         ('year_day',     '',           '')),                          (168,16),   'rsm',   1e-3, None),
                                             ],
                                             columns = ['coefficient',
                                                        'input',
                                                        'nb_intervals',
                                                        'regularization_func',
                                                        'regularization_coef',
                                                        'structure',
                                                        ],
                                             ).set_index(['coefficient', 'input']),
                      ### Sesquivariate model
                      'sesq': pd.DataFrame([
                                                  # Univariate independent features
                                                  ('B', ('temperature',   '',           ''),                            8,       'r2sm', 1e-3, None),
                                                  ('B', ('temperature',   'lag',        pd.DateOffset(hours = 24)),     8,       'r2sm', 1e-5, None),
                                                  ('B', ('temperature',   'maximum',    pd.DateOffset(hours = 24)),     8,       'r2sm', 1e-3, None),
                                                  ('B', ('temperature',   'minimum',    pd.DateOffset(hours = 24)),     8,       'r2sm', 1e-2, None),
                                                  ('B', ('timestamp',     '',           ''),                            'p1',    'rsm',  1e-2, None),
                                                  ('B', ('xmas',          '',           ''),                            'p1',    '',     1e-8, None),
                                                  # Bivariate independent features
                                                  ('B', (('daylight',     '',           ''),
                                                         ('nebulosity',   '',           '')),                          ('p1',2),     'r2sm', 1e-6, None),
                                                  ('B', (('holidays',     '',           ''),
                                                         ('week_hour',    '',           '')),                          ('p1',168),   'r2sm', 1e-6, None),
                                                  ('B', (('holidays',     'lag',        pd.DateOffset(hours = 24)),
                                                         ('week_hour',    '',           '')),                          ('p1',168),   'r2sm', 1e-6, None),
                                                  ('B', (('holidays',     'lag',      - pd.DateOffset(hours = 24)),
                                                         ('week_hour',    '',           '')),                          ('p1',168),   'r2sm', 1e-6, None),
                                                  ('B', (('temperature',  '',           ''),
                                                         ('year_day',     '',           '')),                          (8, 16),      'r2sm', 1e2,  None),
                                                  # Univariate linked features
                                                  ('Cb', ('target',        'lag',        pd.DateOffset(hours = 24)),     2,          'r2sm', 1e-4, None),
                                                  ('Cb', ('week_hour',     '',           ''),                            168,        'rsm',  1e-5, None),
                                                  ('Cb', ('year_day',      '',           ''),                            16,         'r2sm', 1e2,  None),
                                                  # Bivariate linked features
                                                  ('Cbm', (('target',       'lag',       pd.DateOffset(hours = 24)),
                                                           ('week_hour',    '',          '')),                          (2,168),     'rsm',  1e-4, None),
                                                  ('Cbm', (('week_hour',    '',          ''),
                                                           ('year_day',     '',          '')),                          (168,16),    'r2sm', 1e-3, None),
                                             ],
                                             columns = ['coefficient',
                                                        'input',
                                                        'nb_intervals',
                                                        'regularization_func',
                                                        'regularization_coef',
                                                        'structure',
                                                        ],
                                             ).set_index(['coefficient', 'input']),
                      ### Low-rank model
                      'low_rank': pd.DataFrame([
                                                  # Univariate features
                                                  ('B', ('target',        'lag',        pd.DateOffset(hours = 24)),     4,       'r2sm', 1e-4, None),
                                                  ('B', ('temperature',   '',           ''),                            8,       'r2sm', 1e-3, None),
                                                  ('B', ('temperature',   'lag',        pd.DateOffset(hours = 24)),     8,       'r2sm', 1e-5, None),
                                                  ('B', ('temperature',   'maximum',    pd.DateOffset(hours = 24)),     8,       'r2sm', 1e-3, None),
                                                  ('B', ('temperature',   'minimum',    pd.DateOffset(hours = 24)),     8,       'r2sm', 1e-2, None),
                                                  ('B', ('timestamp',     '',           ''),                            'p1',    'rsm',  1e-2, None),
                                                  ('B', ('xmas',          '',           ''),                            'p1',    '',     1e-8, None),
                                                  # Bivariate features
                                                  ('B', (('daylight',     '',           ''),
                                                         ('nebulosity',   '',           '')),                          ('p1',2),  'r2sm',   1e-6, None),
                                                  ('B', (('target',       'lag',        pd.DateOffset(hours = 24)),
                                                         ('week_hour',    '',           '')),                          ('p1',42), 'rsm',    1e-4, None),
                                                  ('B', (('temperature',  '',           ''),
                                                         ('year_day',     '',           '')),                          (8,16),    'r2sm',   1e2,  None),
                                                  # Low-rank univariate features
                                                  ('Blr', ('week_hour',     '',           ''),                            168,     'rsm',   1e-5, None),
                                                  ('Blr', ('year_day',      '',           ''),                            16,      '0r2sm', 1e2,  None),
                                                  # Low-rank bivariate features
                                                  ('B', (('holidays',     '',           ''),
                                                         ('week_hour',    '',           '')),                          ('p1',168), '0r2sm',  1e-6, None),
                                                  ('B', (('holidays',     'lag',        pd.DateOffset(hours = 24)),
                                                         ('week_hour',    '',           '')),                          ('p1',168), '0r2sm',  1e-6, None),
                                                  ('B', (('holidays',     'lag',      - pd.DateOffset(hours = 24)),
                                                         ('week_hour',    '',           '')),                          ('p1',168), '0r2sm',  1e-6, None),
                                                  ('B', (('week_hour',    '',           ''),
                                                         ('year_day',     '',           '')),                          (168,16),   'r2sm',   1e-3, None),
                                             ],
                                             columns = ['coefficient',
                                                        'input',
                                                        'nb_intervals',
                                                        'regularization_func',
                                                        'regularization_coef',
                                                        'structure',
                                                        ],
                                             ).set_index(['coefficient', 'input']),
                      ### Bivariate with low-rank constraints on the bivariate coefficients
                      'interUV': pd.DataFrame([
                                                  # Univariate features
                                                  ('B', ('target',        'lag',        pd.DateOffset(hours = 24)),     4,       'r2sm', 1e-4, None),
                                                  ('B', ('temperature',   '',           ''),                            8,       'r2sm', 1e-3, None),
                                                  ('B', ('temperature',   'lag',        pd.DateOffset(hours = 24)),     8,       'r2sm', 1e-5, None),
                                                  ('B', ('temperature',   'maximum',    pd.DateOffset(hours = 24)),     8,       'r2sm', 1e-3, None),
                                                  ('B', ('temperature',   'minimum',    pd.DateOffset(hours = 24)),     8,       'r2sm', 1e-2, None),
                                                  ('B', ('timestamp',     '',           ''),                            'p1',    'rsm',  1e-2, None),
                                                  ('B', ('week_hour',     '',           ''),                            168,     'rsm',  1e-5, None),
                                                  ('B', ('xmas',          '',           ''),                            'p1',    '',     1e-8, None),
                                                  ('B', ('year_day',      '',           ''),                            16,      'rsm',  1e2,  None),
                                                  # Bivariate unconstrained features
                                                  ('B', (('daylight',     '',           ''),
                                                         ('nebulosity',   '',           '')),                          ('p1',2),    'r2sm',  1e-6, None),
                                                  ('B', (('holidays',     '',           ''),
                                                         ('week_hour',    '',           '')),                          ('p1',168),  'r2sm',  1e-6, None),
                                                  ('B', (('holidays',     'lag',        pd.DateOffset(hours = 24)),
                                                         ('week_hour',    '',           '')),                          ('p1',168),  'r2sm',  1e-6, None),
                                                  ('B', (('holidays',     'lag',      - pd.DateOffset(hours = 24)),
                                                         ('week_hour',    '',           '')),                          ('p1',168),  'r2sm',  1e-6, None),
                                                  # Bivariate low-rank features
                                                  ('Cuv', (('target',       'lag',        pd.DateOffset(hours = 24)),
                                                           ('week_hour',    '',           '')),                        ('p1',42),   'rsm',   1e-4, None),
                                                  ('Cuv', (('temperature',  '',           ''),
                                                           ('year_day',     '',           '')),                        (8,16),      'rsm',   1e2,  None),
                                                  ('Cuv', (('week_hour',    'lag',        pd.DateOffset(hours = 24)),
                                                           ('year_day',     '',           '')),                        (168,16),    'rsm',   1e-3, None),
                                             ],
                                             columns = ['coefficient',
                                                        'input',
                                                        'nb_intervals',
                                                        'regularization_func',
                                                        'regularization_coef',
                                                        'structure',
                                                        ],
                                             ).set_index(['coefficient', 'input']),
                      ### Bivariate with blok-smoothing-splines regularizations
                      'row2sm': pd.DataFrame([
                                                  # Univariate features
                                                  ('B', ('target',        'lag',        pd.DateOffset(hours = 24)),     2,       'row2sm', (1e-4*factor_nb_posts, 1.1), None),
                                                  ('B', ('temperature',   '',           ''),                            8,       'row2sm', (1e-4*factor_nb_posts, 1.1), None),
                                                  ('B', ('temperature',   'lag',        pd.DateOffset(hours = 24)),     8,       'row2sm', (1e-4*factor_nb_posts, 1.1), None),
                                                  ('B', ('temperature',   'maximum',    pd.DateOffset(hours = 24)),     8,       'row2sm', (1e-4*factor_nb_posts, 1.1), None),
                                                  ('B', ('temperature',   'minimum',    pd.DateOffset(hours = 24)),     8,       'row2sm', (1e-4*factor_nb_posts, 1.1), None),
                                                  ('B', ('timestamp',     '',           ''),                            'p1',    'rsm',    1e-3,                        None),
                                                  ('B', ('week_hour',     '',           ''),                            168,     'rsm',    1e-8,                        None),
                                                  ('B', ('xmas',          '',           ''),                            'p1',    '',       1e-8,                        None),
                                                  ('B', ('year_day',      '',           ''),                            16,      'row2sm', 1e-3,                        None),
                                                  # Bivariate features
                                                  ('B', (('daylight',     '',           ''),
                                                         ('nebulosity',   '',           '')),                          ('p1',2),   'row2sm',  (1e-9*factor_nb_posts, 1.1), None),
                                                  ('B', (('holidays',     '',           ''),
                                                         ('week_hour',    '',           '')),                          ('p1',168), 'row2sm',  (1e-9*factor_nb_posts, 1.1), None),
                                                  ('B', (('holidays',     'lag',        pd.DateOffset(hours = 24)),
                                                         ('week_hour',    '',           '')),                          ('p1',168), 'row2sm',  (1e-9*factor_nb_posts, 1.1), None),
                                                  ('B', (('holidays',     'lag',      - pd.DateOffset(hours = 24)),
                                                         ('week_hour',    '',           '')),                          ('p1',168), 'row2sm',  (1e-9*factor_nb_posts, 1.1), None),
                                                  ('B', (('target',       'lag',        pd.DateOffset(hours = 24)),
                                                         ('week_hour',    '',           '')),                          ('p1',42),  'rsm',     1e-6,                        None),
                                                  ('B', (('temperature',  '',           ''),
                                                         ('year_day',     '',           '')),                          (8,16),     'row2sm',  (1e-7*factor_nb_posts, 1.1), None),
                                                  ('B', (('week_hour',    '',           ''),
                                                         ('year_day',     '',           '')),                          (168,16),   'r2sm',    1e-5,                        None),
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
                }

# Choose which variables should be part of the low-rank components and ot the unstructured components
# Blr        for low-rank (over the substations) formulation with a first-order descent algorithm
# Bsp        for independent models and unstructured coedfficients with a first-order descent algorithm
# lbfgs_coef for independent models and unstructured coedfficients with a first-order descent algorithm
# Cuv        for independent models and low-rank interactions
# Cb         for independent models and the univariate part of the sesquivariate constraint
# Cbm        for independent models and the sesquivariate constraint for the interactions

dikt_formula['eCO2mix.administrative_regions', None]      = cp.deepcopy(dikt_formula['eCO2mix.France', None])
dikt_formula['RTE.substations', 'sum']                    = cp.deepcopy(dikt_formula['eCO2mix.France', None])
dikt_formula['RTE.substations', 'administrative_regions'] = cp.deepcopy(dikt_formula['eCO2mix.France', None])
dikt_formula['RTE.substations', 'RTE_regions']            = cp.deepcopy(dikt_formula['eCO2mix.France', None])
dikt_formula['RTE.substations', 'districts']              = cp.deepcopy(dikt_formula['eCO2mix.France', None])
dikt_formula['RTE.quick_test',  None]                     = cp.deepcopy(dikt_formula['RTE.substations', None])
          
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
                          ('RTE.substations','RTE_Regions')            : cp.deepcopy(stopping_criteria['eCO2mix.France',None]),
                          ('RTE.substations','Districts')              : cp.deepcopy(stopping_criteria['eCO2mix.France',None]),
                          })
              