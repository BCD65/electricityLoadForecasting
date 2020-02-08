#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
#
import electricityLoadForecasting.src    as src



def subset_weather(coordinates_weather, coordinates_sites, extra_latitude, extra_longitude):
   
    min_latitude  = coordinates_sites[src.latitude].min() - extra_latitude
    max_latitude  = coordinates_sites[src.latitude].max() + extra_latitude
   
    min_longitude = coordinates_sites[src.longitude].min() - extra_longitude
    max_longitude = coordinates_sites[src.longitude].max() + extra_longitude
    
    cond = True
    for cond2 in [coordinates_weather[src.latitude]  > min_latitude, 
                  coordinates_weather[src.latitude]  < max_latitude, 
                  coordinates_weather[src.longitude] > min_longitude, 
                  coordinates_weather[src.longitude] < max_longitude, 
                  ]:
        cond =np.logical_and(cond, cond2)
    
    subset = coordinates_weather.index[cond]
        
    return subset
            
            
   