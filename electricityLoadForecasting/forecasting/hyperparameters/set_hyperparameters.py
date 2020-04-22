


from .choose_dataset       import choose_dataset
from .choose_inputs        import choose_inputs
from .choose_model         import choose_model
from .set_gam              import set_gam
from .set_afm              import set_afm
from .check_hyperparameters import check_hyperparameters

"""
Set default parameters
Returns a dictionary containing all the parameters
"""

def set_hyperparameters(model = None):

    hprm = {}

    hprm.update(choose_dataset(hprm)) 
    hprm.update(choose_inputs(hprm)) 
    hprm.update(choose_model(hprm, model = model))                
  
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


  


