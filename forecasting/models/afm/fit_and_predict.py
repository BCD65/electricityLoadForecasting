
#
from . import masks
from . import features
from .model import additive_features_model



def fit_and_predict(inputs_training, 
                    Y_training, 
                    inputs_validation,
                    Y_validation,
                    hprm,
                    dikt_assignments,
                    dikt_file_names,
                    ):
    """
    Format targets
    """
    Y_training   = Y_training.values
    Y_validation = Y_validation.values
    
    """
    Masks
    """
    # Compute the masks (indicator of which covariates each site should have access to
    mask_univariate_features = masks.get_mask_univariate(
                                                         hprm,
                                                         inputs_training,
                                                         dikt_assignments,
                                                         file_path = dikt_file_names['features.training.univariate'],
                                                         )
    
    if not hprm['afm.formula'].loc[hprm['afm.formula']['nb_intervals'].apply(lambda x : (type(x) == tuple and len(x) == 2))].empty:
        mask_bivariate_features = masks.get_mask_bivariate(
                                                           hprm,
                                                           inputs_training,
                                                           dikt_assignments,
                                                           file_path = dikt_file_names['features.training.bivariate'],
                                                           ) 
    else:
        mask_bivariate_features = None

        
    """
    Design
    """    
    # training data    
    design      = features.compute_design(inputs_training, 
                                          hprm, 
                                          dikt_file_names, 
                                          db = 'training',
                                          mask_univariate = mask_univariate_features, 
                                          mask_bivariate  = mask_bivariate_features
                                          )
    # validation data
    design.update(features.compute_design (inputs_validation,
                                           hprm,
                                           dikt_file_names, 
                                           dikt_func        = design['dikt_func'], 
                                           mask_univariate  = design.get('mask_univariate'),
                                           dikt_func12      = design.get('dikt_func12'),
                                           mask_bivariate   = design.get('mask_bivariate'),
                                           size2            = design.get('size2'),
                                           size_tensor2     = design.get('size_tensor2'),
                                           db               = 'validation',
                                           ))
    assert len(set(design['X_validation'].keys())) == len(set(design['X_training'].keys())), 'error len X training X validation'

      
    """
    Optimization
    """
    X_training = {k:v 
                  for k, v in design.items() 
                  if 'training' in k
                  }
    X_validation  = {k:v 
                     for k, v in design.items() 
                     if 'validation' in k
                     }
    specs   = {k:v 
               for k, v in design.items() 
               if (    'training'   not in k 
                   and 'validation' not in k)
               }
    model_pred = additive_features_model(hprm,
                                         dikt_file_names = dikt_file_names,
                                         )
    model_pred.fit(specs, 
                   X_training, 
                   Y_training, 
                   X_validation = X_validation, 
                   Y_validation = Y_validation, 
                   )
    
    """
    Predictions
    """
    Y_hat_training   = model_pred.predict(dataset = 'training',    adjust_A = False, verbose = True)
    Y_hat_validation = model_pred.predict(dataset = 'validation',  adjust_A = False, verbose = True)

    
    return design, model_pred, Y_hat_training, Y_hat_validation