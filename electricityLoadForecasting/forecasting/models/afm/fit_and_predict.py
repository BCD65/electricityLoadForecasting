
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
    
    mask_bivariate_features = masks.get_mask_bivariate(
                                                       hprm,
                                                       inputs_training,
                                                       dikt_assignments,
                                                       file_path = dikt_file_names['features.training.bivariate'],
                                                       ) 
        
    """
    Design
    """    
    # training data    
    design      = features.compute_design(inputs_training, 
                                          hprm, 
                                          dikt_file_names, 
                                          mask_univariate = mask_univariate_features, 
                                          mask_bivariate  = mask_bivariate_features,
                                          db              = 'training',
                                          )
    # validation data
    design.update(features.compute_design (inputs_validation,
                                           hprm,
                                           dikt_file_names, 
                                           mask_univariate       = mask_univariate_features,
                                           mask_bivariate        = mask_bivariate_features,
                                           dikt_func             = design['dikt_func'], 
                                           dikt_func12           = design.get('dikt_func12'),
                                           size2                 = design.get('size2'),
                                           size_tensor_bivariate = design.get('size_tensor_bivariate'),
                                           db                    = 'validation',
                                           ))
    assert len(set(design['X1_validation'].keys())) == len(set(design['X1_training'].keys())), 'error len X1 training X1 validation'
    if 'X2_validation' in design:
        assert len(set(design['X2_validation'].keys())) == len(set(design['X2_training'].keys())), 'error len X2 training X2 validation'

      
    """
    Optimization
    """
    X_training = {h:w 
                  for k, v in design.items()
                  for h, w in v.items()
                  if 'training' in k
                  }
    X_validation  = {h:w
                     for k, v in design.items() 
                     for h, w in v.items()
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