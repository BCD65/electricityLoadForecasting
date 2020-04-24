

"""
Definition of indicators to activate/deactivate the download/upload on a remote machine and the load/save on a local machine
Activating all of them should only results in warnings but should not result in a bug
New indicators can be defined for new machines and then activated in the file primitives.py
"""

####################################################################
#              Rules for file transfers                            # 
####################################################################    


###############################################################################
load     = True
save     = True
connect  = False
download = load*connect
upload   = save*connect
delete   = False*save*upload
###############################################################################
save_data           = True*save
save_model          = True*save
save_predictions    = True*save
save_performances   = True*save
###############################################################################
upload_data         = False*upload
upload_model        = False*upload
upload_predictions  = False*upload
upload_performances = False*upload
###############################################################################
load_data           = True*load
load_model          = True*load
load_predictions    = True*load
load_performances   = True*load
###############################################################################
download_data         = False*download
download_model        = False*download
download_predictions  = False*download
download_performances = False*download
###############################################################################



