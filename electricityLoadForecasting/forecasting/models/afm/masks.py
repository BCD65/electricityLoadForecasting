
import numpy as np
import os
#
from electricityLoadForecasting import paths, tools


path_data = os.path.join(paths.outputs,
                         'Data',
                         )

def get_mask_univariate(hprm, inputs, dikt_assignments, file_path = None):
    # Compute the mask for univariate covariates (ie binary indicators to define which covariates each task has access to)
    try:
        mask_univariate = tools.batch_load(
                                           path_data, 
                                           prefix    = file_path,
                                           data_name = 'mask_univariate', 
                                           data_type = 'pickle',
                                           )
    except tools.exceptions.loading_errors:
        mask_univariate = make_mask(inputs,
                                    dikt_assignments,
                                    ) 
        try:
            tools.batch_save(
                             path_data, 
                             prefix    = file_path,
                             data      = mask_univariate,
                             data_name = 'mask_univariate', 
                             data_type = 'pickle',
                             )
        except tools.exceptions.saving_errors:
            pass
    return mask_univariate


def get_mask_bivariate(hprm, inputs, dikt_assignments, file_path = None):
    # Compute the mask for bivariate covariates (ie binary indicators to define which covariates each task has access to)
    # One has to pay attention for instance with the interaction between a weather station in the North of France 
    # and the past loads of a substations in the South : it should not be considered since no substation must access it 
    try:
        mask_bivariate = tools.batch_load(
                                          path_data, 
                                          prefix    = file_path,
                                          data_name = 'mask_bivariate', 
                                          data_type = 'pickle',
                                          )
    except tools.exceptions.loading_errors:
        print('start mask12')
        # An interaction is a scalar product between a left factor and a right factor
        # Compute the masks for each factor
        mask_12  =  make_mask(inputs,
                              dikt_assignments,
                              ) 
        print('finished mask12')
        mask_bivariate = {}
        for (*inpt1, location1) in inputs.columns:
            for (*inpt2, location2) in inputs.columns:
                inpt1 = tuple(inpt1)
                inpt2 = tuple(inpt2)
                if (inpt1,inpt2) in hprm['afm.formula'].index.get_level_values('input'):
                    mask_inter = combine_masks((inpt1, (location1,)),
                                               (inpt2, (location2,)),
                                               mask_12,
                                               )
                    # Combine the masks ie keep only the substtations that have access to both covariates
                    mask_bivariate.update({((inpt1,inpt2),
                                            (location1,location2),
                                            ) : mask_inter,
                                           } 
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
        try:
            tools.batch_save(
                             path_data, 
                             prefix    = file_path,
                             data      = mask_bivariate,
                             data_name = 'mask_bivariate', 
                             data_type = 'pickle',
                             )
        except tools.exceptions.saving_errors:
            pass
    return mask_bivariate


def make_mask(inputs, dikt_assignments):
    print('start make_mask')
    cmask       = {}
    # Get for each substation the order of the other substations 
    # and the weather stations, from the closest to the furthest.
    # This is used to compute masks when it is decided in the parameters 
    # that the X closest substations or weather stations should be considered
    # for each substation
    cmask =  {
              ((name_input,
                transformation,
                parameter,
                ),
               (location,),
                ) : ([ii
                      for ii, name_site in enumerate(dikt_assignments[name_input].columns)
                      if (dikt_assignments[name_input][name_site] == location).any()
                      ]
                     )
              for (name_input, transformation, parameter, location) in inputs.columns
              if (    name_input in dikt_assignments # Input in the assignments
                  and not ((dikt_assignments[name_input] == location).any(axis = 0).all()) # Location is not present for some sites
                  )
              }  
    
    print('end make_mask')
    return cmask

def combine_masks(key1, key2, mask12):
    # Combine the masks of pairs of covariates to compute the masks for the interaction
    if   (key1 not in mask12) and (key2 not in mask12): # Both covariate are accessed by all the substations
        m = slice(None)
    elif (key1     in mask12) and (key2 not in mask12): # One covariate is accessed by all the substations, not the ther
        m = mask12[key1]
    elif (key1 not in mask12) and (key2     in mask12): # One covariate is accessed by all the substations, not the ther
        m = mask12[key2]
    elif (key1     in mask12) and (key2     in mask12): # Both covariates are accessed only by a subset of the substations
        M = set(mask12[key1]) &  set(mask12[key2])
        if M:
            m = np.array(sorted(M))
        else:
            m = slice(0,0) # no substation is interested in this combination
    return m


def cross_mask(mask1,
               mask2,
               ):   
    if bool(mask1) and bool(mask2):
        if (   (type(mask1) == type(slice(None)) and  mask1 == slice(None))
            or (type(mask2) == type(slice(None)) and  mask2 == slice(None))
            ):
            return True
        else:
            return bool(set(mask1) &  set(mask2))
    else:
        return True
 