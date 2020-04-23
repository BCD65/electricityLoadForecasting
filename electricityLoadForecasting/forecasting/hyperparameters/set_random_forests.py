

"""
Parameters for the random forests
"""


def set_random_forests(hprm):
    
    assert hprm['learning.model'] == 'random_forests'
    hprm.update({
                  'random_forests.n_estimators' : 100,
                   })
            
    return hprm

