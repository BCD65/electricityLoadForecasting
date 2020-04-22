#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main functions for the experiments
It successively calls the functions in exp.py
"""

#
import electricityLoadForecasting.paths  as paths
from electricityLoadForecasting.tools                       import exceptions, transcoding
from electricityLoadForecasting.forecasting.experience      import Experience
from electricityLoadForecasting.forecasting.hyperparameters import set_hyperparameters
from electricityLoadForecasting.forecasting.inputs          import load_input_data
import electricityLoadForecasting.forecasting.file_names as file_names
import electricityLoadForecasting.forecasting.models     as models


#%%
#profile
def main(hprm = None, data = None):
    if hprm is None:
        hprm = set_hyperparameters()

    #%%
    print('init')
    exp = Experience(hprm)

    #%%
    print('get_data')
    if data is None:
        exp.data = load_input_data(paths.path_database(hprm['database']))
    else:
        exp.data = data
    exp.select_sites()
    exp.assign_weather()
    exp.unfold_data()
    exp.split_observations()
    exp.dikt_files = file_names.make_dikt_files(exp.hprm,
                                                nb_sites      = len(exp.data['df_sites'].columns),
                                                nb_weather    = len(exp.data['df_weather'].columns.get_level_values(transcoding.user_weather_name)),
                                                dt_training   = exp.target_training.index,
                                                dt_validation = exp.target_validation.index,
                                                )
    #%%
    try:
        print('load_performances')
        exp.load_performances()
#        exp.pre_treatment_adjust()
        print('performances loaded')
    #%%
    except exceptions.loading_errors as e:
        print(repr(e))
        print('performances not loaded')
        try:
            print('load_predictions')
            exp.load_predictions()
#            exp.model, exp.beta, exp.obj_train, exp.obj_test  = ['pred_loaded' for k in range(4)]
            print('predictions loaded')
    #%%
        except exceptions.loading_errors as e:
            print(repr(e))
            print('predictions not loaded')
            #%%
            if models.bool_separated(exp.hprm):
                models.separated(exp)  
            else:
                exp.learning_process()
                #%% 
            try:
                exp.save_predictions()
            except exceptions.saving_errors:
                print('predictions not saved')
        exp.compute_performances()
        try:
            exp.save_performances()
        except exceptions.saving_errors:
            print('performances not saved')
    exp.print_performances()
    #%%    
    #if exp.hprm['any_plot'] and exp.hprm['exp_plot']:
    #    exp.plot_results()
    exp.print_finish()
    return exp

#%%

if __name__ == "__main__":   
    exp  = main()
