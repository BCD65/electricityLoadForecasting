
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
                      ### Univariate
                      'univariate': pd.DataFrame([
                                                  # Univariate features
                                                  ('unconstrained', ('daylight',      '',           ''),                           'p1',    'ridge',         1e-8, None),
                                                  ('unconstrained', ('holidays',      '',           ''),                           'p1',    'ridge',         1e-8, None),
                                                  ('unconstrained', ('holidays',      'lag',        pd.DateOffset(hours = 24)),    'p1',    'ridge',         1e-8, None),
                                                  ('unconstrained', ('holidays',      'lag',      - pd.DateOffset(hours = 24)),    'p1',    'ridge',         1e-8, None),
                                                  ('unconstrained', ('nebulosity',    '',           ''),                           'p1',    'ridge',         1e-8, None),
                                                  ('unconstrained', ('target',        'lag',        pd.DateOffset(hours = 24)),     4,      'smoothing_reg', 1e-4, None),
                                                  ('unconstrained', ('temperature',   '',           ''),                            16,     'smoothing_reg', 1e-4, None),
                                                  ('unconstrained', ('temperature',   'lag',        pd.DateOffset(hours = 24)),     16,     'smoothing_reg', 1e-2, None),
                                                  ('unconstrained', ('temperature',   'maximum',    pd.DateOffset(hours = 24)),     16,     'smoothing_reg', 1e-2, None),
                                                  ('unconstrained', ('temperature',   'minimum',    pd.DateOffset(hours = 24)),     16,     'smoothing_reg', 1e-1, None),
                                                  ('unconstrained', ('timestamp',     '',           ''),                           'p1',    'ridge',         1e-3, None),
                                                  ('unconstrained', ('week_hour',     '',           ''),                            168,    'ridge',         1e-8, None),
                                                  ('unconstrained', ('xmas',          '',           ''),                           'p1',    '',              1e-8, None),
                                                  ('unconstrained', ('year_day',      '',           ''),                            128,    'smoothing_reg', 1e-3, None),
                                              ],
                                             columns = ['coefficient',
                                                        'input',
                                                        'nb_intervals',
                                                        'regularization_func',
                                                        'regularization_coef',
                                                        'structure',
                                                        ],
                                             ).set_index(['coefficient', 'input']),
                      ### Bivariate standard
                      'bivariate': pd.DataFrame([
                                                  # Univariate features
                                                  # ('unconstrained', ('target',        'lag',        pd.DateOffset(hours = 24)),     4,        'smoothing_reg', 1e-4, None),
                                                  ('unconstrained', ('temperature',   '',           ''),                            16,       'smoothing_reg', 1e-4, None),
                                                  ('unconstrained', ('temperature',   'lag',        pd.DateOffset(hours = 24)),     16,       'smoothing_reg', 1e-2, None),
                                                  ('unconstrained', ('temperature',   'maximum',    pd.DateOffset(hours = 24)),     16,       'smoothing_reg', 1e-2, None),
                                                  ('unconstrained', ('temperature',   'minimum',    pd.DateOffset(hours = 24)),     16,       'smoothing_reg', 1e-1, None),
                                                  ('unconstrained', ('timestamp',     '',           ''),                            'p1',     'ridge',         1e-3, None),
                                                  ('unconstrained', ('week_hour',     '',           ''),                            168,      'ridge',         1e-8, None),
                                                  ('unconstrained', ('wind_speed',    '',           ''),                            16,       'smoothing_reg', 1e-5, None),
                                                  ('unconstrained', ('xmas',          '',           ''),                            'p1',     '',              1e-8, None),
                                                  ('unconstrained', ('year_day',      '',           ''),                            128,      'smoothing_reg', 1e-3, None),
                                                  # Bivariate features
                                                  ('unconstrained', (('holidays',     '',           ''),
                                                                     ('week_hour',    '',           '')),                          ('p1',84), 'smoothing_reg', 1e-8, None),
                                                  ('unconstrained', (('holidays',     'lag',        pd.DateOffset(hours = 24)),
                                                                     ('week_hour',    '',           '')),                          ('p1',84), 'smoothing_reg', 1e-8, None),
                                                  ('unconstrained', (('holidays',     'lag',      - pd.DateOffset(hours = 24)),
                                                                     ('week_hour',    '',           '')),                          ('p1',84), 'smoothing_reg', 1e-8, None),
                                                  ('unconstrained', (('daylight',     '',           ''),
                                                                     ('nebulosity',   '',           '')),                          ('p1',2),  'smoothing_reg', 1e-8, None),
                                                  ('unconstrained', (('target',       'lag',        pd.DateOffset(hours = 24)),
                                                                     ('week_hour',    '',           '')),                          (4,84),    'ridge',         1e-6, None),
                                                  ('unconstrained', (('temperature',  '',           ''),
                                                                     ('year_day',     '',           '')),                          (4,32),    'smoothing_reg', 1e-4, None),
                                                  ('unconstrained', (('week_hour',    '',           ''),
                                                                     ('year_day',     '',           '')),                          (168,32),  'smoothing_reg', 1e-5, None),
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
                      'sesquivariate': pd.DataFrame([
                                                  # Univariate independent features
                                                  ('unconstrained', ('temperature',   '',           ''),                            16,        'smoothing_reg', 1e-4, None),
                                                  ('unconstrained', ('temperature',   'lag',        pd.DateOffset(hours = 24)),     16,        'smoothing_reg', 1e-2, None),
                                                  ('unconstrained', ('temperature',   'maximum',    pd.DateOffset(hours = 24)),     16,        'smoothing_reg', 1e-2, None),
                                                  ('unconstrained', ('temperature',   'minimum',    pd.DateOffset(hours = 24)),     16,        'smoothing_reg', 1e-1, None),
                                                  ('unconstrained', ('timestamp',     '',           ''),                            'p1',      'ridge',         1e-3, None),
                                                  ('unconstrained', ('xmas',          '',           ''),                            'p1',      '',              1e-8, None),
                                                  # Bivariate independent features
                                                  ('unconstrained', (('holidays',     '',           ''),
                                                                     ('week_hour',    '',           '')),                          ('p1',84),  'smoothing_reg',  1e-8, None),
                                                  ('unconstrained', (('holidays',     'lag',        pd.DateOffset(hours = 24)),
                                                                     ('week_hour',    '',           '')),                          ('p1',84),  'smoothing_reg',  1e-8, None),
                                                  ('unconstrained', (('holidays',     'lag',      - pd.DateOffset(hours = 24)),
                                                                     ('week_hour',    '',           '')),                          ('p1',84),  'smoothing_reg',  1e-8, None),
                                                  ('unconstrained', (('daylight',     '',           ''),
                                                                     ('nebulosity',   '',           '')),                          ('p1',2),   'smoothing_reg',  1e-8, None),
                                                  ('unconstrained', (('temperature',  '',           ''),
                                                                     ('year_day',     '',           '')),                          (16,128),   'smoothing_reg',  1e-4, None),
                                                  # Univariate linked features
                                                  ('sesquivariate-b', ('target',        'lag',        pd.DateOffset(hours = 24)),     'p1',     'smoothing_reg',  1e-4, None),
                                                  ('sesquivariate-b', ('week_hour',     '',           ''),                            168,      'ridge',          1e-8, None),
                                                  ('sesquivariate-b', ('year_day',      '',           ''),                            128,      'smoothing_reg',  1e-3, None),
                                                  # Bivariate linked features
                                                  ('sesquivariate-bm', (('target',       'lag',       pd.DateOffset(hours = 24)),
                                                                        ('week_hour',    '',          '')),                         ('p1',168), 'ridge',          1e-6, None),
                                                  ('sesquivariate-bm', (('week_hour',    '',          ''),
                                                                        ('year_day',     '',          '')),                         (168,128),  'smoothing_reg',  1e-5, None),
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
                ### Substations level
                ('RTE.substations',
                 None,
                 ) : {
                      ### Univariate
                      'unconstrained-univariate': pd.DataFrame([
                                                  # Univariate features
                                                  ('unconstrained', ('daylight',      '',           ''),                           'p1',    'ridge',         1e-8, None),
                                                  ('unconstrained', ('holidays',      '',           ''),                           'p1',    'ridge',         1e-8, None),
                                                  ('unconstrained', ('holidays',      'lag',        pd.DateOffset(hours = 24)),    'p1',    'ridge',         1e-8, None),
                                                  ('unconstrained', ('holidays',      'lag',      - pd.DateOffset(hours = 24)),    'p1',    'ridge',         1e-8, None),
                                                  ('unconstrained', ('nebulosity',    '',           ''),                           'p1',    'ridge',         1e-8, None),
                                                  ('unconstrained', ('target',        'lag',        pd.DateOffset(hours = 24)),     2,      'smoothing_reg', 1e-4, None),
                                                  ('unconstrained', ('temperature',   '',           ''),                            8,      'smoothing_reg', 1e-3, None),
                                                  ('unconstrained', ('temperature',   'lag',        pd.DateOffset(hours = 24)),     8,      'smoothing_reg', 1e-5, None),
                                                  ('unconstrained', ('temperature',   'maximum',    pd.DateOffset(hours = 24)),     8,      'smoothing_reg', 1e-3, None),
                                                  ('unconstrained', ('temperature',   'minimum',    pd.DateOffset(hours = 24)),     8,      'smoothing_reg', 1e-2, None),
                                                  ('unconstrained', ('timestamp',     '',           ''),                           'p1',    'ridge',         1e-2, None),
                                                  ('unconstrained', ('week_hour',     '',           ''),                            168,    'ridge',         1e-5, None),
                                                  ('unconstrained', ('xmas',          '',           ''),                           'p1',    '',              1e-8, None),
                                                  ('unconstrained', ('year_day',      '',           ''),                            16,     'smoothing_reg', 1e2,  None),
                                              ],
                                             columns = ['coefficient',
                                                        'input',
                                                        'nb_intervals',
                                                        'regularization_func',
                                                        'regularization_coef',
                                                        'structure',
                                                        ],
                                             ).set_index(['coefficient', 'input']),
                      ### Bivariate standard
                      'unconstrained-bivariate': pd.DataFrame([
                                                  # Univariate features
                                                  ('unconstrained', ('target',        'lag',        pd.DateOffset(hours = 24)),     2,       'smoothing_reg', 1e-4, None),
                                                  ('unconstrained', ('temperature',   '',           ''),                            8,       'smoothing_reg', 1e-3, None),
                                                  ('unconstrained', ('temperature',   'lag',        pd.DateOffset(hours = 24)),     8,       'smoothing_reg', 1e-5, None),
                                                  ('unconstrained', ('temperature',   'maximum',    pd.DateOffset(hours = 24)),     8,       'smoothing_reg', 1e-3, None),
                                                  ('unconstrained', ('temperature',   'minimum',    pd.DateOffset(hours = 24)),     8,       'smoothing_reg', 1e-2, None),
                                                  ('unconstrained', ('timestamp',     '',           ''),                            'p1',    'ridge',         1e-2, None),
                                                  ('unconstrained', ('week_hour',     '',           ''),                            168,     'ridge',         1e-5, None),
                                                  ('unconstrained', ('xmas',          '',           ''),                            'p1',    '',              1e-8, None),
                                                  ('unconstrained', ('year_day',      '',           ''),                            16,      'smoothing_reg', 1e2,  None),
                                                  # Bivariate features
                                                  ('unconstrained', (('daylight',     '',           ''),
                                                                     ('nebulosity',   '',           '')),                          ('p1',2),   'smoothing_reg',  1e-6, None),
                                                  ('unconstrained', (('holidays',     '',           ''),
                                                                     ('week_hour',    '',           '')),                          ('p1',168), 'smoothing_reg',  1e-6, None),
                                                  ('unconstrained', (('holidays',     'lag',        pd.DateOffset(hours = 24)),
                                                                     ('week_hour',    '',           '')),                          ('p1',168), 'smoothing_reg',  1e-6, None),
                                                  ('unconstrained', (('holidays',     'lag',      - pd.DateOffset(hours = 24)),
                                                                     ('week_hour',    '',           '')),                          ('p1',168), 'smoothing_reg',  1e-6, None),
                                                  ('unconstrained', (('target',       'lag',        pd.DateOffset(hours = 24)),
                                                                     ('week_hour',    '',           '')),                          ('p1',42),  'ridge',          1e-4, None),
                                                  ('unconstrained', (('temperature',  '',           ''),
                                                                     ('year_day',     '',           '')),                          (8,16),     'smoothing_reg',  1e2,  None),
                                                  ('unconstrained', (('week_hour',    '',           ''),
                                                                     ('year_day',     '',           '')),                          (168,16),   'ridge',          1e-3, None),
                                             ],
                                             columns = ['coefficient',
                                                        'input',
                                                        'nb_intervals',
                                                        'regularization_func',
                                                        'regularization_coef',
                                                        'structure',
                                                        ],
                                             ).set_index(['coefficient', 'input']),
                      ### Sesquivariate 
                      'sesquivariate': pd.DataFrame([
                                                  # Univariate independent features
                                                  ('unconstrained', ('temperature',   '',           ''),                            8,         'smoothing_reg', 1e-3, None),
                                                  ('unconstrained', ('temperature',   'lag',        pd.DateOffset(hours = 24)),     8,         'smoothing_reg', 1e-5, None),
                                                  ('unconstrained', ('temperature',   'maximum',    pd.DateOffset(hours = 24)),     8,         'smoothing_reg', 1e-3, None),
                                                  ('unconstrained', ('temperature',   'minimum',    pd.DateOffset(hours = 24)),     8,         'smoothing_reg', 1e-2, None),
                                                  ('unconstrained', ('timestamp',     '',           ''),                            'p1',      'ridge',         1e-2, None),
                                                  ('unconstrained', ('xmas',          '',           ''),                            'p1',      '',              1e-8, None),
                                                  # Bivariate independent features
                                                  ('unconstrained', (('daylight',     '',           ''),
                                                                     ('nebulosity',   '',           '')),                          ('p1',2),   'smoothing_reg', 1e-6, None),
                                                  ('unconstrained', (('holidays',     '',           ''),
                                                                     ('week_hour',    '',           '')),                          ('p1',168), 'smoothing_reg', 1e-6, None),
                                                  ('unconstrained', (('holidays',     'lag',        pd.DateOffset(hours = 24)),
                                                                     ('week_hour',    '',           '')),                          ('p1',168), 'smoothing_reg', 1e-6, None),
                                                  ('unconstrained', (('holidays',     'lag',      - pd.DateOffset(hours = 24)),
                                                                     ('week_hour',    '',           '')),                          ('p1',168), 'smoothing_reg', 1e-6, None),
                                                  ('unconstrained', (('temperature',  '',           ''),
                                                                     ('year_day',     '',           '')),                          (8, 16),    'smoothing_reg', 1e2,  None),
                                                  # Univariate linked features
                                                  ('sesquivariate-b', ('target',        'lag',        pd.DateOffset(hours = 24)),     2,        'smoothing_reg', 1e-4, None),
                                                  ('sesquivariate-b', ('week_hour',     '',           ''),                            168,      'ridge',         1e-5, None),
                                                  ('sesquivariate-b', ('year_day',      '',           ''),                            16,       'smoothing_reg', 1e2,  None),
                                                  # Bivariate linked features
                                                  ('sesquivariate-bm', (('target',       'lag',       pd.DateOffset(hours = 24)),
                                                                        ('week_hour',    '',          '')),                          (2,168),   'ridge',         1e-4, None),
                                                  ('sesquivariate-bm', (('week_hour',    '',          ''),
                                                                        ('year_day',     '',          '')),                          (168,16),  'smoothing_reg', 1e-3, None),
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
                      'low-rank': pd.DataFrame([
                                                  # Univariate features
                                                  ('unconstrained', ('target',        'lag',        pd.DateOffset(hours = 24)),     4,        'smoothing_reg',        1e-4, None),
                                                  ('unconstrained', ('temperature',   '',           ''),                            8,        'smoothing_reg',        1e-3, None),
                                                  ('unconstrained', ('temperature',   'lag',        pd.DateOffset(hours = 24)),     8,        'smoothing_reg',        1e-5, None),
                                                  ('unconstrained', ('temperature',   'maximum',    pd.DateOffset(hours = 24)),     8,        'smoothing_reg',        1e-3, None),
                                                  ('unconstrained', ('temperature',   'minimum',    pd.DateOffset(hours = 24)),     8,        'smoothing_reg',        1e-2, None),
                                                  ('unconstrained', ('timestamp',     '',           ''),                           'p1',      'ridge',                1e-2, None),
                                                  ('unconstrained', ('xmas',          '',           ''),                           'p1',      '',                     1e-8, None),
                                                  # Bivariate features
                                                  ('unconstrained', (('daylight',     '',           ''),
                                                                     ('nebulosity',   '',           '')),                         ('p1',2),   'smoothing_reg',        1e-6, None),
                                                  ('unconstrained', (('target',       'lag',        pd.DateOffset(hours = 24)),
                                                                     ('week_hour',    '',           '')),                         ('p1',42),  'ridge',                1e-4, None),
                                                  ('unconstrained', (('temperature',  '',           ''),
                                                                     ('year_day',     '',           '')),                         (8,16),     'smoothing_reg',        1e2,  None),
                                                  # Low-rank univariate features
                                                  ('low-rank-UVt',  ('week_hour',     '',           ''),                           168,       'ridge',                1e-5, 4),
                                                  ('low-rank-UVt',  ('year_day',      '',           ''),                           16,        'factor_smoothing_reg', 1e2,  4),
                                                  # Low-rank bivariate features
                                                  ('unconstrained', (('holidays',     '',           ''),
                                                                     ('week_hour',    '',           '')),                         ('p1',168), 'factor_smoothing_reg',  1e-6, None),
                                                  ('unconstrained', (('holidays',     'lag',        pd.DateOffset(hours = 24)),
                                                                     ('week_hour',    '',           '')),                         ('p1',168), 'factor_smoothing_reg',  1e-6, None),
                                                  ('unconstrained', (('holidays',     'lag',      - pd.DateOffset(hours = 24)),
                                                                     ('week_hour',    '',           '')),                         ('p1',168), 'factor_smoothing_reg',  1e-6, None),
                                                  ('unconstrained', (('week_hour',    '',           ''),
                                                                     ('year_day',     '',           '')),                         (168,16),   'smoothing_reg',         1e-3, None),
                                             ],
                                             columns = ['coefficient',
                                                        'input',
                                                        'nb_intervals',
                                                        'regularization_func',
                                                        'regularization_coef',
                                                        'structure',
                                                        ],
                                             ).set_index(['coefficient', 'input']),
                      ### Tensor product
                      'tensor-product': pd.DataFrame([
                                                  # Univariate features
                                                  ('unconstrained', ('target',        'lag',        pd.DateOffset(hours = 24)),     4,         'smoothing_reg', 1e-4, None),
                                                  ('unconstrained', ('temperature',   '',           ''),                            8,         'smoothing_reg', 1e-3, None),
                                                  ('unconstrained', ('temperature',   'lag',        pd.DateOffset(hours = 24)),     8,         'smoothing_reg', 1e-5, None),
                                                  ('unconstrained', ('temperature',   'maximum',    pd.DateOffset(hours = 24)),     8,         'smoothing_reg', 1e-3, None),
                                                  ('unconstrained', ('temperature',   'minimum',    pd.DateOffset(hours = 24)),     8,         'smoothing_reg', 1e-2, None),
                                                  ('unconstrained', ('timestamp',     '',           ''),                            'p1',      'ridge',         1e-2, None),
                                                  ('unconstrained', ('week_hour',     '',           ''),                            168,       'ridge',         1e-5, None),
                                                  ('unconstrained', ('xmas',          '',           ''),                            'p1',      '',              1e-8, None),
                                                  ('unconstrained', ('year_day',      '',           ''),                            16,        'ridge',         1e2,  None),
                                                  # Bivariate unconstrained features
                                                  ('unconstrained', (('daylight',     '',           ''),
                                                                     ('nebulosity',   '',           '')),                          ('p1',2),   'smoothing_reg', 1e-6, None),
                                                  ('unconstrained', (('holidays',     '',           ''),
                                                                     ('week_hour',    '',           '')),                          ('p1',168), 'smoothing_reg', 1e-6, None),
                                                  ('unconstrained', (('holidays',     'lag',        pd.DateOffset(hours = 24)),
                                                                     ('week_hour',    '',           '')),                          ('p1',168), 'smoothing_reg', 1e-6, None),
                                                  ('unconstrained', (('holidays',     'lag',      - pd.DateOffset(hours = 24)),
                                                                     ('week_hour',    '',           '')),                          ('p1',168), 'smoothing_reg', 1e-6, None),
                                                  # Bivariate low-rank features
                                                  ('tensor-product-L.R', (('target',       'lag',        pd.DateOffset(hours = 24)),
                                                                          ('week_hour',    '',           '')),                        ('p1',42),  'ridge',         1e-4, 2),
                                                  ('tensor-product-L.R', (('temperature',  '',           ''),
                                                                          ('year_day',     '',           '')),                        (8,16),     'ridge',         1e2,  2),
                                                  ('tensor-product-L.R', (('week_hour',    'lag',        pd.DateOffset(hours = 24)),
                                                                          ('year_day',     '',           '')),                        (168,16),   'ridge',         1e-3, 2),
                                             ],
                                             columns = ['coefficient',
                                                        'input',
                                                        'nb_intervals',
                                                        'regularization_func',
                                                        'regularization_coef',
                                                        'structure',
                                                        ],
                                             ).set_index(['coefficient', 'input']),
                      ### Block-smoothing-splines regularizations
                      'block_smoothing_reg': pd.DataFrame([
                                                  # Univariate features
                                                  ('unconstrained', ('target',        'lag',        pd.DateOffset(hours = 24)),     2,         'block_smoothing_reg', (1e-4*factor_nb_posts, 1.1), None),
                                                  ('unconstrained', ('temperature',   '',           ''),                            8,         'block_smoothing_reg', (1e-4*factor_nb_posts, 1.1), None),
                                                  ('unconstrained', ('temperature',   'lag',        pd.DateOffset(hours = 24)),     8,         'block_smoothing_reg', (1e-4*factor_nb_posts, 1.1), None),
                                                  ('unconstrained', ('temperature',   'maximum',    pd.DateOffset(hours = 24)),     8,         'block_smoothing_reg', (1e-4*factor_nb_posts, 1.1), None),
                                                  ('unconstrained', ('temperature',   'minimum',    pd.DateOffset(hours = 24)),     8,         'block_smoothing_reg', (1e-4*factor_nb_posts, 1.1), None),
                                                  ('unconstrained', ('timestamp',     '',           ''),                            'p1',      'ridge',                1e-3,                       None),
                                                  ('unconstrained', ('week_hour',     '',           ''),                            168,       'ridge',                1e-8,                       None),
                                                  ('unconstrained', ('xmas',          '',           ''),                            'p1',      '',                     1e-8,                       None),
                                                  ('unconstrained', ('year_day',      '',           ''),                            16,        'block_smoothing_reg',  1e-3,                       None),
                                                  # Bivariate features
                                                  ('unconstrained', (('daylight',     '',           ''),
                                                         ('nebulosity',   '',           '')),                          ('p1',2),   'block_smoothing_reg',  (1e-9*factor_nb_posts, 1.1), None),
                                                  ('unconstrained', (('holidays',     '',           ''),
                                                         ('week_hour',    '',           '')),                          ('p1',168), 'block_smoothing_reg',  (1e-9*factor_nb_posts, 1.1), None),
                                                  ('unconstrained', (('holidays',     'lag',        pd.DateOffset(hours = 24)),
                                                         ('week_hour',    '',           '')),                          ('p1',168), 'block_smoothing_reg',  (1e-9*factor_nb_posts, 1.1), None),
                                                  ('unconstrained', (('holidays',     'lag',      - pd.DateOffset(hours = 24)),
                                                         ('week_hour',    '',           '')),                          ('p1',168), 'block_smoothing_reg',  (1e-9*factor_nb_posts, 1.1), None),
                                                  ('unconstrained', (('target',       'lag',        pd.DateOffset(hours = 24)),
                                                         ('week_hour',    '',           '')),                          ('p1',42),  'ridge',                 1e-6,                       None),
                                                  ('unconstrained', (('temperature',  '',           ''),
                                                         ('year_day',     '',           '')),                          (8,16),     'block_smoothing_reg',  (1e-7*factor_nb_posts, 1.1), None),
                                                  ('unconstrained', (('week_hour',    '',           ''),
                                                         ('year_day',     '',           '')),                          (168,16),   'smoothing_reg',         1e-5,                       None),
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
# B          for independent models and unstructured coedfficients with a first-order descent algorithm
# Cuv        for independent models and low-rank interactions
# Cb         for independent models and the univariate part of the sesquivariate constraint
# Cbm        for independent models and the sesquivariate constraint for the interactions

dikt_formula['eCO2mix.administrative_regions', None]      = cp.deepcopy(dikt_formula['eCO2mix.France',  None])
dikt_formula['RTE.substations', 'sum']                    = cp.deepcopy(dikt_formula['eCO2mix.France',  None])
dikt_formula['RTE.substations', 'administrative_regions'] = cp.deepcopy(dikt_formula['eCO2mix.France',  None])
dikt_formula['RTE.substations', 'RTE_regions']            = cp.deepcopy(dikt_formula['eCO2mix.France',  None])
dikt_formula['RTE.substations', 'districts']              = cp.deepcopy(dikt_formula['eCO2mix.France',  None])
dikt_formula['RTE.quick_test',  None]                     = cp.deepcopy(dikt_formula['RTE.substations', None])
          
###############################################################################
###                       Stopping Criteria                                 ###
###############################################################################
              
stopping_criteria = {
                     ('eCO2mix.France',
                      None,
                      ) : {
                           'afm.algorithm.lbfgs.maxfun'  : 1e8,
                           'afm.algorithm.lbfgs.maxiter' : 1e8,                             
                           'afm.algorithm.lbfgs.tol'     : 1e-10,
                           'afm.algorithm.lbfgs.pgtol'   : 1e-10,
                           },
                     ('eCO2mix.administrative_regions',
                      None,
                      ) : {
                           'afm.algorithm.lbfgs.maxfun'  : 1e8,
                           'afm.algorithm.lbfgs.maxiter' : 1e8,                             
                           'afm.algorithm.lbfgs.pgtol'   : 1e-9,
                           'afm.algorithm.lbfgs.tol'     : 1e-9,
                           },
                    ('RTE.substations',
                     None,
                      ) : {
                           'afm.algorithm.lbfgs.maxfun'  : 1e8,
                           'afm.algorithm.lbfgs.maxiter' : 1e8,                             
                           'afm.algorithm.lbfgs.tol'     : 1e-6,
                           'afm.algorithm.lbfgs.pgtol'   : 1e-6,
                           },
                     }

stopping_criteria.update({
                          ('RTE.substations','sum')                    : cp.deepcopy(stopping_criteria['eCO2mix.France',  None]),
                          ('RTE.substations','Administrative_regions') : cp.deepcopy(stopping_criteria['eCO2mix.France',  None]),
                          ('RTE.substations','RTE_Regions')            : cp.deepcopy(stopping_criteria['eCO2mix.France',  None]),
                          ('RTE.substations','Districts')              : cp.deepcopy(stopping_criteria['eCO2mix.France',  None]),
                          ('RTE.quick_test', 'Districts')              : cp.deepcopy(stopping_criteria['RTE.substations', None]),
                          })
              