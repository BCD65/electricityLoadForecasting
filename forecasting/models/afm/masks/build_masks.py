#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pickle
import re
import os
#
import electricityLoadForecasting.paths as paths
import electricityLoadForecasting.tools as tools

#
#import itertools
#import primitives
#import build_data
#from custom_exception import custex
#from subprocess import CalledProcessError, TimeoutExpired


#tupleErrors = (CalledProcessError, custex, TimeoutExpired, OSError, AssertionError)


def get_mask_univariate(hprm, inputs, dikt_assignments, file_path = None):
    # Compute the mask for univariate covariates (ie binary indicators to define which covariates each task has access to)
    #prefix1 = dikt['primitives_train1']
    try:
        mask_univariate = tools.batch_load(
                                           os.path.join(paths.outputs,
                                                        'Saved/Masks',
                                                        ), 
                                           prefix    = file_path,
                                           data_name = 'mask_univariate', 
                                           data_type = 'dictionary',
                                           )
    except tools.exceptions.loading_errors:
        mask_univariate = make_mask(
                                    inputs,
                                    dikt_assignments,
                                    ) 
        try:
            tools.batch_save(
                             os.path.join(paths.outputs,
                                          'Saved/Masks',
                                          ), 
                             prefix    = file_path,
                             data      = mask_univariate,
                             data_name = 'mask_univariate', 
                             data_type = 'dictionary',
                             )
        except tools.exceptions.saving_errors:
            pass
    return mask_univariate


def get_mask_bivariate(hprm, inputs, dikt_assignments, file_path = None):
    # Compute the mask for bivariate covariates (ie binary indicators to define which covariates each task has access to)
    # One has to pay attention for instance with the interaction between a weather station in the North of France 
    # and the past loads of a substations in the South : it should not be considered since no substation must access it 
    #prefix2 = dikt['primitives_train2']
    try:
        mask_bivariate = tools.batch_load(
                                          os.path.join(paths.outputs,
                                                       'Saved/Masks',
                                                       ), 
                                          prefix    = file_path,
                                          data_name = 'mask_bivariate', 
                                          data_type = 'dictionary',
                                          )
    except tools.exceptions.loading_errors:
        mask_bivariate = build_mask_bivariate(inputs, dikt_assignments)  
        try:
            tools.batch_save(
                             os.path.join(paths.outputs,
                                          'Saved/Masks',
                                          ), 
                             prefix    = file_path,
                             data      = mask_bivariate,
                             data_name = 'mask_bivariate', 
                             data_type = 'dictionary',
                             )
        except tools.exceptions.saving_errors:
            pass
    return mask_bivariate


#def build_mask_univariate(inputs_training, assignments):
#    # Mask1 
#    # Discard the covariates that are not in param['selected_variables']
#    univariate_keys1 = {
#                        k 
#                        for k in data.keys() 
#                        if param['data_cat'][k] in param['selected_variables']
#                        }
#    cmask1     = make_mask(inputs_training, 
#                           param, 
#                           prm_mask, 
#                           coor_posts, 
#                           coor_stations, 
#                           )
#    return cmask1


def build_mask_bivariate(param, data, coor_posts, coor_stations):
    print('start build_mask2')
    #posts_names = param['posts_names']
    # Mask2   
    print('start mask12')
    # An interaction is a scalar product between a left factor and a right factor
    # Compute the masks for each factor
    mask_12  =  make_mask(inputs, dikt_assignments) 
    #mask12right = build_mask1(param, param['tf_masks'], data, coor_posts, coor_stations)
    print('finished mask12')
    cmask2 = {}
    bivariate_keys = set([
                          bivkey 
                          for coef, list_keys in param['tf_config_coef'].items() 
                          for bivkey in list_keys 
                          if '#' in bivkey
                          ])
    for ii, keyleft in enumerate(data):
        for jj, keyright in enumerate(data):
            # Browse all tthe combinations
            print('\r'+str(jj), '/', len(data), ' - ', ii, '/', len(data), end = '')
            actual_cpl = param['data_cat'][keyleft ]+'#'+param['data_cat'][keyright] in bivariate_keys # Indicator that the interaction is considered
            if actual_cpl:
                # Combine the masks ie keep only the substtations that have access to both covariates
                mask_inter = combine_masks(keyleft, mask12left, keyright, mask12right, posts_names)
                cmask2.update({keyleft+'#'+keyright:mask_inter} 
                              if  (not (    type(mask_inter) == slice 
                                        and mask_inter == slice(0,0)
                                        )
                                   and  (   type(mask_inter) != slice 
                                         or mask_inter != slice(None)
                                         )
                                   )
                              else
                              {}
                              ) 
    # For each interaction keyleft+'#'+keyright, the corresponding value in cmask2 is the list of the substations that have access to it
    # If the interaction is not in cmask2, then all substations have access to it.
    print('\nend build_mask2')
    return cmask2


def cross_mask(maska, k_a, maskb, k_b):
    key_a = '#'.join(sorted(k_a.split('#')))    
    key_b = '#'.join(sorted(k_b.split('#')))    
    if key_a in maska and key_b in maskb:
        if (   (type(maska[key_a]) == type(slice(None)) and  maska[key_a] == slice(None))
            or (type(maskb[key_b]) == type(slice(None)) and  maskb[key_b] == slice(None))
            ):
            return True
        else:
            return bool(set(maska[key_a]) &  set(maskb[key_b])) # Relevant product
    else:
        return True # One has no mask ie is for all posts


def make_mask(inputs, dikt_assignments):
    print('start make_mask')
    cmask       = {}
    # Get for each substation the order of the other substations and the weather stations, from the closest to the furthest
    # This is used to compute masks when it is decided in the parameters that the X closest substations or weather stations should be considered
    # for each substation
    # Filter keys and entities
#    for ii, (name_input, transformation, parameter, location) in enumerate(inputs.columns):
        #if cat in prm_mask:
#        if 'meteo' in cat or 'nebu' in cat:
#            # Choose the order in terms of distance for the weather stations
#            rolling_away = rolling_away_stations
#            names        = prm['stations_names']
#        elif cat == 'targetlag':
#            # Choose the order in terms of distance for the substations
#            rolling_away = rolling_away_posts
#            names        = prm['posts_names']
#        else:
#            continue
#        rolling_away = {post:list(filter(lambda x : x in names, rollin)) 
#                        for post, rollin in rolling_away.items() 
#                        if post in posts_names
#                        }
#        for jj, key in enumerate(data_keys):
#            if prm['data_cat'][key] == cat:
#                print('\r{0:5}'.format(ii), ' / ', '{0:5}'.format(len(prm['data_cat'])), 
#                      ' - ', 
#                      '{0:5}'.format(jj), ' / ', '{0:5}'.format(len(data_keys)), 
#                       end = '', 
#                       )
                #re.sub(r'[0-9]+', '', key):
                #name    = names[int(re.findall(r'\d+', key)[0])]
#        mask_ow = dikt_assignments[name_input][location]
##        np.array([jj
##                            for jj, e in enumerate(posts_names) 
##                            if name in rolling_away.get(e,[])[:prm_mask[cat]] # Keep the (prm_mask[cat]) closest
##                            ]).astype(int)
#        if len(mask_ow) != len(posts_names):
#            cmask.update({key : mask_ow})
                # otherwise the covariate is accessed by all substations and is not added to the cmask
                
    cmask =  {
              (name_input,
               #transformation,
               #parameter,
               location,
               ) : ([name_site
                     for name_site in dikt_assignments[name_input].columns
                     if (dikt_assignments[name_input][name_site] == location).any()
                     ]
                    if (    name_input in dikt_assignments # Input in the assignments
                        and not ((dikt_assignments[name_input] == location).any(axis = 0).all()) # Location not present for all the site
                        )
                    else
                    slice(None)
                    )
              for (name_input, transformation, parameter, location) in inputs.columns
              }           
    
    print('\nend make_mask')
    return cmask

def combine_masks(keyleft, mask12left, keyright, mask12right, posts_names):
    # Combine the masks of pairs of covariates to compute the masks for the interaction
    if   (keyleft not in mask12left) and (keyright not in mask12right): # Both covariate are accessed by all the substations
        m = slice(None)
    elif (keyleft     in mask12left) and (keyright not in mask12right): # One covariate is accessed by all the substations, not the ther
        m = mask12left[keyleft]
    elif (keyleft not in mask12left) and (keyright     in mask12right): # One covariate is accessed by all the substations, not the ther
        m = mask12right[keyright]
    elif (keyleft     in mask12left) and (keyright     in mask12right): # Both covariates are accessed only by a subset of the substations
        M = set(mask12left[keyleft]) &  set(mask12right[keyright])
        if M:
            m = np.array(sorted(M))#.astype(int)
            if m.shape == len(posts_names):
                m = slice(None)
        else:
            m = slice(0,0) # no substation is interested in this combination
    return m


 