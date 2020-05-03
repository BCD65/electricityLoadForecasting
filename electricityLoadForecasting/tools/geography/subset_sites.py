

import numpy as np
#
from electricityLoadForecasting import paths
from .aggregation_sites         import get_dikt_districts, get_dikt_regions_admin, get_dikt_regions_rte
from ..                         import transcoding



def subset_sites(coordinates_sites, zone):
   
    if   zone == 'all': # all the sites
        subset = slice(None)
    
    elif zone == 'Paris': # the substations in a rectangle around Paris
        subset = subsample_Paris(coordinates_sites)
    
    elif zone == '4_in_Paris': # A few substations in Paris (mainly for testing)
        subset = subsample_4_in_Paris(coordinates_sites)
        
    elif zone == 'Normandie':# substations in Normandie + Paris
        subset = subsample_Normandie(coordinates_sites.index)
        
    else:
        subset = subsample_region(coordinates_sites.index,
                                  zone,
                                  )
        
    return subset
            

def subsample_Normandie(list_sites):
    with open(paths.extras + 'Postes_Normandie_Paris.npy', 'rb') as f_normandie:
        posts_normandie = np.load(f_normandie)
    subset = sorted(set(list_sites).intersection(posts_normandie))
    return subset


def subsample_Paris(coordinates_sites):
    min_latitude  = 48.7
    max_latitude  = 49
    min_longitude = 2.1
    max_longitude = 2.6
    subset        = [k for k in coordinates_sites.index
                     if  (coordinates_sites[transcoding.longitude][k] >= min_longitude) 
                     &   (coordinates_sites[transcoding.longitude][k] <= max_longitude) 
                     &   (coordinates_sites[transcoding.latitude] [k] >= min_latitude)  
                     &   (coordinates_sites[transcoding.latitude] [k] <= max_latitude) 
                     ]
    return subset


def subsample_4_in_Paris(coordinates_sites):
    subset = [k 
                for ii, k in enumerate(subsample_Paris(coordinates_sites)) 
                if ii < 4
                ]
    return subset


def subsample_region(list_sites,
                     zone,
                     ):
    dikt_regions_admin = get_dikt_regions_admin()
    dikt_regions_rte   = get_dikt_regions_rte()
    dikt_districts     = get_dikt_districts()
    if zone in dikt_regions_admin:
        dikt = dikt_regions_admin
    elif zone in dikt_regions_rte:
        dikt = dikt_regions_rte
    elif zone in dikt_districts:
        dikt = dikt_districts
    else:
        raise ValueError
    subset = [k
              for k in list_sites
              if dikt[k] == zone
              ]
    assert len(subset) > 0
    return subset

