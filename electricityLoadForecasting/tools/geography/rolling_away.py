

import numpy as np
import pandas as pd


def haversine(theta):
    return (1 - np.cos(theta))/2

def rolling_away(coorA, coorB):
    # For each element in coorA, 
    # it sorts the element in coorB 
    # from the closest to the furthest
    assert type(coorA) == pd.DataFrame
    assert type(coorB) == pd.DataFrame
    radian_coorA = coorA*np.pi/180
    radian_coorB = coorB*np.pi/180
    latA  = radian_coorA[['latitude']].values
    latB  = radian_coorB[['latitude']].values.T
    longA = radian_coorA[['longitude']].values
    longB = radian_coorB[['longitude']].values.T
    distances    = pd.DataFrame((  haversine(latA  - latB)
                                 + haversine(longA - longB)*np.cos(latA)*np.cos(latB)
                                 ), 
                                index   = radian_coorA.index,
                                columns = radian_coorB.index,
                                )
    rolling_away = {nameA : [x for _,x in sorted(zip(distances.loc[nameA], distances.columns))]
                    for nameA in distances.index
                    }
    return rolling_away#, rolling_away_ind

