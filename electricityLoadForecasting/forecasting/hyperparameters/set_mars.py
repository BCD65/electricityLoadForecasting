

"""
Parameters for mars
"""


def set_mars(hprm):
    
    assert hprm['learning.model'] == 'mars'
    hprm.update({
                 'mars.verbose' : 1,
                 'mars.thresh'  : 0.00001,
                 })
    
    return hprm

