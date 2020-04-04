

#from .rules_transfer       import rules_transfer
#from .partial_computations import partial_computations
from .choose_dataset       import choose_dataset
from .choose_inputs        import choose_inputs
from .choose_model         import choose_model
from .set_gam              import set_gam
from .set_afm              import set_afm
#from .select_variables     import select_variables
#from .set_approx_tf        import set_approx_tf
#from .pre_treatment        import pre_treatment
#from .post_treatment       import post_treatment
#from .rules_plot           import rules_plot
#from .extra_parameters     import extra_parameters

from .check_hyperparameters import check_hyperparameters

"""
Set default parameters
Returns a dictionary containing all the parameters
"""

def set_hyperparameters(model = None):

    hprm = {}

#    param.update(rules_transfer())         
#    param.update(partial_computations())
    hprm.update(choose_dataset(hprm)) 
    hprm.update(choose_inputs(hprm)) 
    hprm.update(choose_model(hprm, model = model))                
#    param.update(select_variables(param))  
#    param.update(rules_plot()) 
#    param.update(extra_parameters()) 
#
#    if param['method'] == 'approx_tf':
#        param.update(set_approx_tf(param))
#        param.update(set_self_reg())
#    
    if   hprm['learning.model'] == 'afm':
        hprm.update(set_afm(hprm))
   
    elif hprm['learning.model'] == 'gam':
        hprm.update(set_gam(hprm))
           
    elif hprm['learning.model'] in {'random_forests', 'regression_tree', 'xgboost', 'svr'}:
        pass
    
    else:
        raise ValueError
    
    hprm = check_hyperparameters(hprm)
    
    return hprm 


  

