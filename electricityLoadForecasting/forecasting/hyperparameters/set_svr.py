

"""
Parameters for the svr model
"""


def set_svr(hprm):
    
    assert hprm['learning.model'] == 'svr'
    hprm.update({
                 'svr.C'       : 1,
                 'svr.epsilon' : 1e-3,
                  })
            
    return hprm