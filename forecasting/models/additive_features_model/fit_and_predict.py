

from . import masks



def fit_and_predict(inputs_training, 
                    Y_training, 
                    inputs_validation,
                    hprm,
                    dikt_assignments,
                    dikt_files,
                    ):
    """
    Masks
    """
    # Compute the masks (indicator of which substations should access which covariates)
    mask_univariate = masks.get_mask_univariate(
                                                hprm,
                                                inputs_training,
                                                dikt_assignments,
                                                file_path = dikt_files['experience.modeling'],
                                                ) 
    if bool([e
             for coef, list_keys in param['tf_config_coef'].items()
             for e in list_keys
             if '#'     in e
             ]):
        cmask2 = masks.get_mask2(
                                     hprm,
                                     inputs_training,
                                     file_path = dikt_files['experience.modeling'],
                                     ) 
    else:
        cmask2 = None
        
    """
    Design
    """    
    # training data    
    design      = approx_tf.compute_design(data_train, 
                                           param, 
                                           path_Saved + 'Data/', 
                                           dikt, 
                                           db    = 'train',
                                           given_mask1 = mask_univariate, 
                                           given_mask2 = cmask2
                                           )
    # validation data
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

      
    """
    Optimization
    """
    model_pred       = approx_tf.tf_reg(param, path_Saved, dikt, 
                                        target_train, target_test, 
                                        design,
                                        )
    
    """
    Predictions
    """
    Y_hat_training   = model_pred.predict(data = 'train', adjust_A = False, verbose = True)
    Y_hat_validation = model_pred.predict(data = 'test',  adjust_A = False, verbose = True)
    
    ans_kw.update({
                   'design'       : design,
                   })