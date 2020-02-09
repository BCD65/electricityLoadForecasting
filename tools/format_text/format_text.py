

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