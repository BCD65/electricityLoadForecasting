
"""
Hyperparameters for GAM
"""

def set_gam(hprm):    
    
    hprm.update({
                 'gam.univariate_functions'   : {
                                                 #'do'             : ('', ''),
                                                 #'h'              : ('s', 'bs = "cc"'), 
                                                 'temperature'     : ('s', ''), 
                                                 'temperature_max' : ('s', ''), 
                                                 'temperature_min' : ('s', ''), 
                                                 'temperature_dif' : ('s', ''), 
                                                 'temperature_smo' : ('s', ''), 
                                                 'target_lag'      : ('s', ''), 
                                                 'week_hour'       : ('s', 'k = 7, bs = "cc"'), 
                                                 #'yd'             : ('s', ''), 
                                                 #'stamp'           : ('', ''), 
                                                 },
                 'gam.bivariate_functions'    : {
                                                 #('h', 'wd')          : ('ti', 'bs = c("cc", "tp"), k = c(24, 7)'),
                                                 #('h', 'yd')          : ('ti', 'bs = c("cc", "tp")'), 
                                                 #('wd', 'yd')         : ('ti', 'bs = c("tp", "tp")'), 
                                                 #('targetlag', 'wd')  : ('ti', 'bs = c("tp", "tp")'), 
                                                 },
                   })
    return hprm

##gam_selected_variables
#selected_variables = (
#                      'dado',
#                      'dbdo',
#                      'dl',
#                      'do',
#                      'h',
#                      'meteo',
#                      'meteolag',
#                      'meteomax',
#                      'meteomin', 
#                      'meteosmo', 
#                      'nebu',
#                      'ones',
#                      'stamp',
#                      'targetlag',
#                      #'wh',
#                      'wd',
#                      'wd_bin',
#                      'we',
#                      'xmas',
#                      'yd',
#                      #'meteodif',
#                      #'meteosmo', 
#                      #'sbrk',
#                      #'wd',
#                      )