



transcoding_weather = {' ' : '-',
                       'Marseille-Marignane' : 'Marignane',
                       'Le-Luc'              : 'Le Luc',
                       'St-'                 : 'Saint-',
                       'Bordeaux-Merignac'   : 'Bordeaux',
                       'Caen-Carpiquet'      : 'Caen',
                       'Clermont-Fd'         : 'Clermont-Ferrand',
                       'Dijon-Longvic'       : 'Dijon',
                       'Lille-Lesquin'       : 'Lille',
                       'Lyon-Saint-Exupery'  : 'Lyon-Satolas',
                       'Nancy-Ochey'         : 'Nancy-Essey',
                       'Nantes-Bouguenais'   : 'Nantes',
                       'Rennes-Saint-Jacques': 'Rennes',
                       'Strasbourg-Entzheim' : 'Strasbourg',
                       'Tarbes-Ossun'        : 'Tarbes-Ossuns',
                       "Dumont-D'Urville"    : 'Trappes',
                       'Troyes-Barberey'     : 'Troyes',
                       }

transcoding_sites   = {' ' : '-',
                       '.' : '_',
                       }


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