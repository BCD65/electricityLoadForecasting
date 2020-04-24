

import subprocess 
import pickle


loading_errors = (
                  AssertionError, 
                  AttributeError, 
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
