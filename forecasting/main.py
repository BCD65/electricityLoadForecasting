#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Main functions for the experiments
It successively calls the functions in exp.py
"""

#import numpy as np
#import copy as cp
#import pickle
#import datetime as dt
#from termcolor import colored
#
from electricityLoadForecasting.tools                       import exceptions
from electricityLoadForecasting.forecasting.experience      import Experience
from electricityLoadForecasting.forecasting.hyperparameters import set_hyperparameters
from electricityLoadForecasting.forecasting.inputs          import load_input_data
from electricityLoadForecasting.forecasting.meta_model      import meta_model
#import post_treatment;  importlib.reload(post_treatment)
#import primitives;      importlib.reload(primitives)
#import str_make;        importlib.reload(str_make)
#import munging;         importlib.reload(munging)


# Ignore warnings
#import warnings
#warnings.filterwarnings("ignore")


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
        exp.data = load_input_data()
    else:
        exp.data = data
    exp.select_sites()
    exp.assign_weather()
    exp.unfold_data()
    exp.split_observations()
    #exp.file_names = str_make.mk_dic(exp.hprm)
    #exp.set_attributes()
    #%%
    try:
        raise NotImplementedError
#        print('load_perf')
#        exp.load_performances()
#        exp.prediction_train, exp.prediction_test         = ['perf_loaded' for k in range(2)]
#        exp.model, exp.beta, exp.obj_train, exp.obj_test  = ['perf_loaded' for k in range(4)]
#        exp.pre_treatment_adjust()
#        print('perf loaded')
#        exp.print_performances()
    #%%
    except exceptions.loading_errors as e:
        print(e)
        print('performances not loaded')
        try:
            raise NotImplementedError
#            print('load_pred')
#            exp.load_predictions()
#            exp.model, exp.beta, exp.obj_train, exp.obj_test  = ['pred_loaded' for k in range(4)]
#            print('pred loaded')
    #%%
        except exceptions.loading_errors as e:
            print(e)
            print('predictions not loaded')
            #%%
            if meta_model.bool_meta_model(exp.hprm):
                meta_model(exp)  
            else:
                exp.learning_process()
                
                #%% 
            try:
                exp.save_predictions()
            except exceptions.saving_errors:
                pass
        exp.compute_performances()
        try:
            exp.save_performances()
        except exceptions.saving_errors:
            pass
        exp.print_performances()
        
    #%%    
    #if exp.hprm['any_plot'] and exp.hprm['exp_plot']:
    #    exp.plot_results()
    exp.print_finish()
    return exp

#%%

if __name__ == "__main__":   
    exp  = main()
