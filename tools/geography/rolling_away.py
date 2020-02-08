#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import numpy as np
import pandas as pd
from termcolor import colored


def haversine(theta):
    return (1 - np.cos(theta))/2


def rolling_away(coorA, coorB):
    print(colored('add try load and save', 'red', 'on_yellow'))
    """
    For each element in coorA, it sorts the element in coorB from the closest to the furthest
    """
    assert type(coorA) == pd.DataFrame
    assert type(coorB) == pd.DataFrame
    lenA    = coorA.shape[0]
    lenB    = coorB.shape[0]
    namesA  = list(coorA.index)
    namesB  = list(coorB.index)
    hav_AB  = np.zeros((lenA, lenB))
    rolling_away      = {}
    #rolling_away_ind  = {}
    for i in range(hav_AB.shape[0]):
        for j in range(hav_AB.shape[1]):
            longA, latA = [x*np.pi/180 for x in coorA.loc[namesA[i]]]
            longB, latB = [x*np.pi/180 for x in coorB.loc[namesB[j]]]
            hav_AB[i,j] = haversine(latA - latB) + np.cos(latA)*np.cos(latB)*haversine(longA - longB)
        rolling_away    [namesA[i]] = [x for _,x in sorted(zip(hav_AB[i], namesB))]
        #rolling_away_ind[namesA[i]] = [k for _,k in sorted(zip(hav_AB[i], np.arange(lenB)))] 
    return rolling_away#, rolling_away_ind

