"""
Definition of a class for the experiment
An instance containes all the parameters, the input data, the model, the prediction (after the optimization) etc.
"""

import numpy    as np
import pandas   as pd
import datetime as dt
import os
from termcolor import colored

#
import electricityLoadForecasting.paths  as paths
import electricityLoadForecasting.src    as src
import electricityLoadForecasting.tools  as tools
from .. import performances, hyperparameters, inputs
#
import electricityLoadForecasting.forecasting.benchmarks as benchmarks
import electricityLoadForecasting.forecasting.config     as config

#import munging;                 importlib.reload(munging)
#import parameters;              importlib.reload(parameters)
#import post_treatment;          importlib.reload(post_treatment)
#import pre_treatment;           importlib.reload(pre_treatment)
#import str_make;                importlib.reload(str_make)
#import nb_format;               importlib.reload(nb_format)
#import performances;            importlib.reload(performances)
#import misc;                    importlib.reload(misc)
#import weights_computation;     importlib.reload(weights_computation)
#import naive_models;            importlib.reload(naive_models)
#import primitives;              importlib.reload(primitives)
#import regression;              importlib.reload(regression)
#import approx_tf;               importlib.reload(approx_tf)
#import build_data;              importlib.reload(build_data)
####
#from .plot_data import plot_data
#from .plot_pred                import plot_pred
#from .plot_1d_norm_residuals   import plot_1d_norm_residuals
#from .plot_1d_marginal         import plot_1d_marginal
#from .plot_covariance_residual import plot_covariance_residual
#import archives_plot


class Experience(object):

###############################################################################
    
    def __init__(self, hprm):
        self.dt_start = dt.datetime.now()
        print('Start : {0}'.format(self.dt_start.strftime("%Y-%m-%d %H:%M:%S")))
        self.hprm     = hyperparameters.check_hyperparameters(hprm)
    
    def print_finish(self):
        self.dt_end = dt.datetime.now()
        self.total_time = self.dt_end - self.dt_start
        hours, remainder = divmod(self.total_time.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        print('Total : ', '{:02} hours {:02} minutes {:02} seconds'.format(int(hours), int(minutes), int(seconds)))


###############################################################################
  
    def select_sites(self, ):   
        self.nb_input_sites = self.data['df_sites'].shape[1]
        
        # Discard sites in TRASH_SITES
        self.data['df_sites']            .drop(labels = self.hprm['sites.trash'], axis = 1, inplace = True)
        self.data['df_coordinates_sites'].drop(labels = self.hprm['sites.trash'], axis = 0, inplace = True)
        
        # Subsample a region
        subset_sites = tools.geography.subset_sites(self.data['df_sites'].columns, self.hprm['sites.zone'])
        self.data['df_sites']             = self.data['df_sites']                [subset_sites]
        self.data['df_coordinates_sites'] = self.data['df_coordinates_sites'].loc[subset_sites]
        
        # Aggregate
        mapping = tools.geography.aggregation_sites(self.data['df_sites'].columns, self.hprm['sites.aggregation'])
        self.data['df_sites']             = self.data['df_sites']            .groupby(by = mapping, axis = 1).agg(np.sum)
        self.data['df_coordinates_sites'] = self.data['df_coordinates_sites'].groupby(by = mapping, axis = 0).agg(np.mean)
        
        # Compute distance-related orders
        self.rolling_away_sites = tools.geography.rolling_away(self.data['df_coordinates_sites'], 
                                                               self.data['df_coordinates_sites'], 
                                                               )
        # Assign weather stations
        self.assignment_sites = pd.DataFrame({k:v[:self.hprm['inputs.nb_sites_per_site']]
                                              for k, v in self.rolling_away_sites.items()
                                              })

###############################################################################

    def assign_weather(self, ):

        # Choose weather source
        self.data['df_weather'] = self.data['df_weather'].xs(key   = self.hprm['weather.source'],
                                                             axis  = 1,
                                                             level = src.user_source,
                                                             )
        
        # Restrict the weather stations to a rectangle defined by the sites
        subset_weather = tools.geography.subset_weather(self.data['df_coordinates_weather'],
                                                        self.data['df_coordinates_sites'],
                                                        self.hprm['weather.extra_latitude'],
                                                        self.hprm['weather.extra_longitude'],
                                                        )
        self.data['df_weather']             = self.data['df_weather']                [subset_weather]
        self.data['df_coordinates_weather'] = self.data['df_coordinates_weather'].loc[subset_weather]        
        
        # Aggregate weather stations
        self.data['df_weather'], self.data['df_coordinates_weather'] = tools.geography.aggregation_weather(self.data['df_weather'], 
                                                                                                           self.data['df_coordinates_weather'], 
                                                                                                           self.hprm['weather.aggregation'],
                                                                                                           )
        # Compute distance-related orders
        self.rolling_away_weather = tools.geography.rolling_away(self.data['df_coordinates_sites'], 
                                                                 self.data['df_coordinates_weather'], 
                                                                 )
        # Assign weather stations
        self.assignment_weather_stations = pd.DataFrame({k:v[:self.hprm['inputs.nb_weather_per_site']]
                                                         for k, v in self.rolling_away_weather.items()
                                                         })
        self.data['df_weather'] = self.data['df_weather'][sorted(np.unique(self.assignment_weather_stations))]

###############################################################################

    def unfold_data(self, ):
        self.data['df_calendar'] = tools.calendar.compute_calendar_variables(self.data['df_sites'].index)
        basket_original_data     = {**{key : self.data['df_calendar'][[key]] for key in self.data['df_calendar']} , 
                                    **{physical_quantity : self.data['df_weather'].xs(key = physical_quantity, axis = 1, level = src.user_physical_quantity)
                                       for physical_quantity in self.data['df_weather'].columns.get_level_values(src.user_physical_quantity)
                                       },
                                    'target' : self.data['df_sites'], 
                                    }
        self.assignments = {'target' : self.assignment_sites,
                            **{physical_quantity : self.assignment_weather_stations
                               for physical_quantity in self.data['df_weather'].columns.get_level_values(src.user_physical_quantity)
                               },
                            }

        self.inputs = {}
        for name_input, transformation, parameter in self.hprm['inputs.selection']:
            self.inputs[name_input, transformation, parameter] = inputs.transform_input(basket_original_data[name_input], 
                                                                                        transformation,
                                                                                        parameter,
                                                                                        )
        self.target    = self.data['df_sites']
  
###############################################################################

    def split_observations(self, ):
        self.dt_training, self.dt_validation = performances.split_population(self.hprm, 
                                                                             freq = pd.infer_freq(self.target.index),
                                                                             tz   = self.target.index.tz,
                                                                             )
        self.inputs_training   = {k : v.loc[self.dt_training]   for k, v in self.inputs.items()}
        self.inputs_validation = {k : v.loc[self.dt_validation] for k, v in self.inputs.items()}
        self.target_training   = self.target.loc[self.dt_training]
        self.target_validation = self.target.loc[self.dt_validation]



    def load_performances(self):
        # Load performances ie r2, MAPE, NMSE etc.
        if not config.load_performances:
            raise IOError('fail on purpose load performances')
        self.performances = tools.batch_load(os.path.join(paths.outputs,
                                                          'Saved/Performances',
                                                          ), 
                                             self.dikt_files['exp'], 
                                             data_name = 'performances', 
                                             data_type = 'dictionary',
                                             )

    #profile
    def load_predictions(self):
        # Load the predictions 
        if not config.load_predictions:
            raise IOError('fail on purpose load predictions')
        self.predictions = tools.batch_load(os.path.join(paths.outputs,
                                                         'Saved/Predictions',
                                                         ), 
                                            self.dikt_files['exp'], 
                                            data_name = 'predictions', 
                                            data_type = 'dictionary',
                                            )
        self.target_training       = self.predictions['target_training']
        self.prediction_training   = self.predictions['prediction_training']
        self.target_validation     = self.predictions['target_validation']
        self.prediction_validation = self.predictions['prediction_validation']

            
    #profile
    def learning_process(self, ):   
        print('learning_process')
        
        # normalize
        self.target_mean  = self.target_training.mean(axis = 0)
        self.Y_training   = (self.target_training   - self.target_mean)/self.target_mean
        self.Y_validation = (self.target_validation - self.target_mean)/self.target_mean  
        
        
        if self.hprm['learning.method'] == 'additive_features_model': # Standard bivariate linear model - the main focus of our work
            #path_Saved  = param['path_outputs'] + 'Saved/' 
            assert len(data_train) == len(data_test)
            # Compute the masks (indicator of which substations should access which covariates)
            cmask1 = approx_tf.get_mask1(param, data_train, dikt, self.coor_posts, self.coor_stations, path = path_Saved + 'Data/') 
            if bool([e for coef, list_keys in param['tf_config_coef'].items() for e in list_keys if '#'     in e]):
                cmask2 = approx_tf.get_mask2(param, data_train, dikt, self.coor_posts, self.coor_stations, path = path_Saved + 'Data/') 
            else:
                cmask2 = None
            if not param['just_get_tf_coef']:
                assert len(data_train) == len(data_test)
                # Compute the design for the training and the test set
                design      = approx_tf.compute_design(data_train, 
                                                       param, 
                                                       path_Saved + 'Data/', 
                                                       dikt, 
                                                       db    = 'train',
                                                       given_mask1 = cmask1, 
                                                       given_mask2 = cmask2
                                                       )
                design.update(approx_tf.compute_design (data_test, param, 
                                                        path_Saved+ 'Data/', dikt, 
                                                        dikt_func        = design['dikt_func'], 
                                                        given_mask1      = design.get('mask1'),
                                                        dikt_func12      = design.get('dikt_func12'),
                                                        given_mask2      = design.get('mask2'),
                                                        size2            = design.get('size2'),
                                                        size_tensor2     = design.get('size_tensor2'),
                                                        db               = 'test',
                                                        ))
                assert len(set(design['X_test'].keys())) == len(set(design['X_train'].keys())), 'error len X train X test'
            if param['just_build_features'] : 
                return 0    
            # Call the module for optimization
            model_pred       = approx_tf.tf_reg(param, path_Saved, dikt, 
                                                target_train, target_test, 
                                                design,
                                                )
            self.Y_hat_training   = model_pred.predict(data = 'train', adjust_A = False, verbose = True)
            self.Y_hat_validation = model_pred.predict(data = 'test',  adjust_A = False, verbose = True)
            ans_kw.update({
                           'design'       : design,
                           })
            
        
        elif self.hprm['learning.method'] == 'gam': # Gam models
            self.Y_hat_training, self.Y_hat_validation = benchmarks.gam.fit_and_predict(self.inputs_training, 
                                                                                        self.Y_training, 
                                                                                        self.inputs_validation,
                                                                                        self.hprm,
                                                                                        self.assignments,
                                                                                        )
            self.model = None
            
            
        elif self.hprm['learning.method'] in {'random_forests', 'regression_tree', 'xgboost', 'svr'}:
            self.Y_hat_training, self.Y_hat_validation, self.model = benchmarks.classical.fit_and_predict(self.inputs_training, 
                                                                                                          self.Y_training, 
                                                                                                          self.inputs_validation,
                                                                                                          self.hprm,
                                                                                                          self.assignments,
                                                                                                          )
        else:
            raise ValueError
            
        # de-normalize
        self.prediction_training   = self.Y_hat_training   * self.target_mean + self.target_mean
        self.prediction_validation = self.Y_hat_validation * self.target_mean + self.target_mean 
        
            
    def save_predictions(self):
        if not config.save_predictions:
            raise IOError('perf not saved on purpose')
        self.prediction_dikt = {
                                'target_training'       : self.target_training, 
                                'prediction_training'   : self.prediction_training, 
                                'target_validation'     : self.target_validation,
                                'prediction_validation' : self.prediction_validation,
                                'weather'               : self.data['df_weather']
                                }
        tools.batch_save(os.path.join(paths.outputs, 
                                      'Saved/Predictions',
                                      ), 
                         prefix    = self.dikt_files['exp'], 
                         data      = self.prediction_dikt, 
                         data_name = 'predictions', 
                         data_type = 'dictionary',
                         )
             
            
    def compute_performances(self):       
        """
        Performances
        """
        self.performances = {
                             'sites_names'   : self.target_training.columns,
                             'weather_names' : self.data['df_weather'].columns.get_level_values(level = src.user_weather_name),
                             'dt_training'   : self.dt_training,
                             'dt_validation' : self.dt_validation,
                             'model'         : {},
                             'aggregated'    : {},
                             'good_days'     : {},
                             'lag_168'       : {},
                             }
        # Model performances
        self.performances['model']['training']   = performances.get_performances(self.target_training, 
                                                                                 self.prediction_training,
                                                                                 )
        self.performances['model']['validation'] = performances.get_performances(self.target_validation,
                                                                                 self.prediction_validation,
                                                                                 )
        # Model performances on the aggregated load
        self.performances['aggregated']['training']   = performances.get_performances(self.target_training.sum(axis=1)[:,None],
                                                                                      self.prediction_training.sum(axis=1)[:,None],
                                                                                      )
        self.performances['aggregated']['validation'] = performances.get_performances(self.target_validation.sum(axis=1)[:,None],
                                                                                      self.prediction_validation.sum(axis=1)[:,None],
                                                                                      )
        # Model performances on good_days
        self.good_days = ~(self.data['df_calendar']['holidays'] + self.data['df_calendar']['xmas']).astype(bool)
        self.performances['good_days']['training']   = performances.get_performances(self.target_training[self.good_days[self.dt_training]],
                                                                                     self.prediction_training[self.good_days[self.dt_training]],
                                                                                     )
        self.performances['good_days']['validation'] = performances.get_performances(self.target_validation[self.good_days[self.dt_validation]],
                                                                                     self.prediction_validation[self.good_days[self.dt_validation]],
                                                                                     )
        # Naive benchmark model
        self.performances['lag_168']['training']   = performances.get_performances(self.target_training.loc[self.target_training.index.min() + pd.DateOffset(hours = 168):],
                                                                                   self.target_training.set_index(self.target_training.index + pd.DateOffset(hours = 168)).loc[:self.target_training.index.max()],
                                                                                   )
        self.performances['lag_168']['validation'] = performances.get_performances(self.target_validation.loc[self.target_validation.index.min() + pd.DateOffset(hours = 168):],
                                                                                   self.target_validation.set_index(self.target_validation.index + pd.DateOffset(hours = 168)).loc[:self.target_validation.index.max()],
                                                                                   )

    def save_performances(self,):   
        if not config.save_performances:
            raise IOError('performances not saved on purpose')
        tools.batch_save(os.path.join(paths.outputs,
                                      'Saved/Performances',
                                      ), 
                         prefix    = self.dikt_files['exp'], 
                         data      = self.performances, 
                         data_name = 'performances', 
                         data_type = 'dictionary',
                         )
        
        
    def print_performances(self):
        print('Exp : ' + self.dikt_files['exp'].replace('/', '/\n\t'))
        print()
        print('Quantiles MAPE TEST : ')
        performances.print_quantiles(self.performances['model']['validation']['mapes'])
        print()
        
        print('LAG168 : ')
        performances.print_performances(self.performances['lag_168']['validation'])
        print('AGGREGATED : ')
        performances.print_performances(self.performances['aggregated']['validation'])
        print('MODEL : ')
        performances.print_performances(self.performances['model']['validation'])


        
#    def plot_results(self):        
#        print(colored('BEGIN PLOTS', 'green'))
#        try:
#            if self.hprm.get('exp_plot_1d_effect') and self.hprm['method'] == 'approx_tf':
#                archives_plot.plot_1d_effect(self)
#            if self.hprm.get('exp_plot_1d_effect_sum') and self.hprm['method'] == 'approx_tf':
#                archives_plot.plot_1d_effect_sum(self)
#            if self.hprm.get('exp_plot_1d_marginal'):
#                plot_1d_marginal(self)
#            if self.hprm.get('exp_plot_covariance_residuals'):
#                plot_covariance_residual(self)
#            if self.hprm.get('exp_plot_1d_norm_residuals'):
#                plot_1d_norm_residuals(self)
#            if self.hprm.get('exp_plot_pred'):
#                plot_pred(self)
#        except Exception as e:
#            print(colored('\n\n' + str(e) + '\n', 'red', 'on_cyan'))  




