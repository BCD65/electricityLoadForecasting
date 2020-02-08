

import subprocess 
import pickle



class custex(Exception):
     
    def __init__(self, value):
         self.value = value
     
    def __str__(self):
         return repr(self.value)
     

loading_errors = tuple_errors = (
                                 AttributeError, 
                                 FileNotFoundError, 
                                 AssertionError, 
                                 OSError, 
                                 KeyError, 
                                 EOFError,
                                 subprocess.TimeoutExpired, 
                                 subprocess.CalledProcessError, 
                                 pickle.UnpicklingError,
                                 custex,  
                                 )
