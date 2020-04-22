


import numpy as np

def round_nb(x, nb_app = 1):
    a = round(x, 
              nb_app -int(np.floor(np.log10(max(1e-12, np.abs(x))))),
              )
    return a
