
import re
#
import electricityLoadForecasting.src as src


def format_weather_station_name(name):
    name = name.title()
    for k, v in src.transcoding_weather.items():
        name = name.replace(k,v)
    return name


def format_site_name(name):
    name = name.upper()
    for k, v in src.transcoding_sites.items():
        name = name.replace(k,v)
    return name


def ltx_trsfrm(s):
    # Transform string for compatibility with plots in Latex
    if '_nolegend_' in s:
        t = '_nolegend_'
    else:
        t = s.replace('_', ' ').replace('#', 'XX')
    return t


def format_file_names(s):
    dikt_replace = {
                    ' '         : '_',
                    ','         : '',
                    ']'         : '',
                    '['         : '',
                    '\''        : '',
                    '.'         : 'd',
                    '-'         : 'm',
                    '('         : '_',
                    ')'         : '_',
                    '+'         : 'p',
                    '//'        : '/',
                    #'__'        : '_',
                    'meteosmo'  : 'ms',
                    'meteodif'  : 'md',
                    'meteomin'  : 'mi',
                    'meteomax'  : 'ma',
                    'meteolag'  : 'ml',
                    'meteo'     : 'm',
                    'nebu'      : 'n',
                    'targetlag' : 'tl',
                    'daylight'  : 'dl',
                    'approx_tf' : 'tf', 
                    'filter'    : 'fl', 
                    }
    for key, value in dikt_replace.items():
        s = s.replace(key, value)
    v = re.sub(r'__*', '_', s)
    v = re.sub('(_/)+', '/', v)
    return v.lower()