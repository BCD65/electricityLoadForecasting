
"""
Hyperparameters for GAM
"""

def set_gam(hprm):    
    
    hprm.update({
                 'gam.univariate_functions'   : {
                                                 #'holidays'       : ('', ''),
                                                 #'hour'           : ('s', 'bs = "cc"'), 
                                                 'temperature'     : ('s', ''), 
                                                 'temperature_max' : ('s', ''), 
                                                 'temperature_min' : ('s', ''), 
                                                 'temperature_dif' : ('s', ''), 
                                                 'temperature_smo' : ('s', ''), 
                                                 'target_lag'      : ('s', ''), 
                                                 'week_hour'       : ('s', 'k = 7, bs = "cc"'), 
                                                 #'year_day'       : ('s', ''), 
                                                 #'timestamp'      : ('', ''), 
                                                 },
                 'gam.bivariate_functions'    : {
                                                 #('hour',      'week_day')  : ('ti', 'bs = c("cc", "tp"), k = c(24, 7)'),
                                                 #('hour',      'year_day')  : ('ti', 'bs = c("cc", "tp")'), 
                                                 #('week_day',  'year_day')  : ('ti', 'bs = c("tp", "tp")'), 
                                                 #('targetlag', 'week_day')  : ('ti', 'bs = c("tp", "tp")'), 
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