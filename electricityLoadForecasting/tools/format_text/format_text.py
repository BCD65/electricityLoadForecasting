
import re
import os
#
from ..transcoding import transcoding_sites, transcoding_weather


def format_weather_station_name(name):
    name = name.title()
    for k, v in transcoding_weather.items():
        name = name.replace(k,v)
    return name


def format_site_name(name):
    name = name.upper()
    for k, v in transcoding_sites.items():
        name = name.replace(k,v)
    return name


def ltx_trsfrm(s):
    # Transform string for compatibility with plots in Latex
    if '_nolegend_' in s:
        t = '_nolegend_'
    else:
        t = s.replace('_', ' ').replace('#', 'XX')
    return t


def format_file_names(txt):
    s = txt.lower()
    dikt_replace = {
                    ### standard caracters
                    ','              : '',
                    "'"              : '',
                    '<'              : '',
                    '>'              : '',
                    ':'              : '',
                    '='              : '',
                    '\*'             : '',
                    ' '              : '_',
                    '-'              : 'm',
                    'administrative' : 'adm',
                    'eco2mix'        : 'e2m',
                    'binary'         : 'bin',
                    'dateoffset'     : '',
                    'daylight'       : 'dl',
                    'holidays'       : 'hld',
                    'hours'          : 'h',
                    'maximum'        : 'max',
                    'minimum'        : 'min',
                    'none'           : '',
                    'nebulosity'     : 'neb',
                    'regions'        : 'reg',
                    'smoothing'      : 'sm',
                    'target'         : 'tg',
                    'temperature'    : 'temp', 
                    'training'       : 'train', 
                    'unconstrained'  : 'unc',
                    'validation'     : 'val',
                    'week_day'       : 'wd',
                    'week_hour'      : 'wh',
                    'year_day'       : 'yd',
                    
                    ### special caracters
                    '\]'          : '',
                    '\['          : '',
                    '\('          : '_',
                    '\)'          : '_',
                    '\.'          : 'd',
                    '\+'          : 'p',
                    '\*'          : '',
                    
                    ### regex
                    '_+'                     : '_',
                    '(_{0})+'.format(os.sep) : os.sep,
                    '{0}_*'.format(os.sep)   : os.sep,
                    }
    for key, value in dikt_replace.items():
        s = re.sub(key, value, s)
    return s



