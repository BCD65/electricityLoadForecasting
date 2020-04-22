


"""
Parameters for GAM
"""



def set_gam(hprm):    
    
    hprm.update({
                 'gam.univariate_functions'   : {
                                                 #'h'              : ('s', 'bs = "cc"'), 
                                                 'week_hour'       : ('s', 'k=7, bs = "cc"'), 
                                                 #'yd'             : ('s', ''), 
                                                 #'meteo'          : ('s', ''), 
                                                 'temperature'     : ('s', ''), 
                                                 'temperature_max' : ('s', ''), 
                                                 'temperature_min' : ('s', ''), 
                                                 'temperature_dif' : ('s', ''), 
                                                 'temperature_smo' : ('s', ''), 
                                                 'target_lag'      : ('s', ''), 
                                                 #'do'             : ('', ''),
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
