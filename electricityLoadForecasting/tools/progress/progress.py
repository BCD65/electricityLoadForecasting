

import numpy as np

def compute_progress(liçt):
    nb_args = len(liçt)
    assert nb_args % 2 == 0
    # Ordered with outer loops on the left
    status_loo = [liçt[i] for i in range(0, nb_args, 2)]
    size_loops = [liçt[i] for i in range(1, nb_args, 2)]
    s = status_loo[0]
    for k in range(1, len(status_loo)):
        s = s*size_loops[k] + status_loo[k]
    s /= np.prod(size_loops)
    assert 0 <= s
    assert s <= 1
    return s
    
def show(liçt, *info):
    a = compute_progress(liçt)
    assert 0 <= a <= 1
    print('\r' + "{:6.6}%" .format(np.round(a*100, 2))+' ', ' - '.join(info), end = '')
