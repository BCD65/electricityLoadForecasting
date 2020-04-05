

import subprocess 
import pickle



#class custex(Exception):
#     
#    def __init__(self, value):
#         self.value = value
#     
#    def __str__(self):
#         return repr(self.value)
     

loading_errors = tuple_errors = (
                                 AssertionError, 
                                 AttributeError, 
                                 #custex,  
                                 EOFError,
                                 FileNotFoundError, 
                                 KeyError, 
                                 NotImplementedError,
                                 OSError, 
                                 #OverflowError,
                                 pickle.UnpicklingError,
                                 subprocess.CalledProcessError, 
                                 subprocess.TimeoutExpired, 
                                 )

saving_errors = loading_errors
