


from .check_hyperparameters import check_hyperparameters
from .choose_dataset        import choose_dataset
from .choose_inputs         import choose_inputs
from .choose_model          import choose_model
from .set_afm               import set_afm
from .set_gam               import set_gam
from .set_mars              import set_mars
from .set_random_forests    import set_random_forests
from .set_svr               import set_svr

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
   
    elif hprm['learning.model'] == 'mars':
        hprm.update(set_mars(hprm))
   
    elif hprm['learning.model'] == 'random_forests':
        hprm.update(set_random_forests(hprm))
   
    elif hprm['learning.model'] == 'svr':
        hprm.update(set_svr(hprm))
           
    elif hprm['learning.model'] in {'regression_tree', 'xgboost'}:
        pass
    
    else:
        raise ValueError('Incorrect learning.model in hprm')
    
    hprm = check_hyperparameters(hprm)
    
    return hprm 


  


