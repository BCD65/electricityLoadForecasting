
import os
import pickle
import pandas as pd
#
import electricityLoadForecasting.paths as paths
from .. import format_text


def aggregation_sites(list_sites, aggregation_level):
    if aggregation_level is None:
        dikt = list_sites
    elif aggregation_level == 'sum':
        dikt = {k : 'sum' for k in list_sites}
    elif aggregation_level == 'administrative_regions':
        dikt = get_dikt_regions_admin()
    elif aggregation_level == 'RTE_regions':
        dikt = get_dikt_regions_rte()
    elif aggregation_level == 'districts':
        dikt = get_dikt_districts()
    else:
        raise ValueError
    return dikt


def get_dikt_districts():
    with open(os.path.join(paths.extras,
                           'dikt_districts.pkl',
                           ),
              'rb') as f: 
        dikt_districts = pickle.load(f)
    return dikt_districts


def get_dikt_regions_admin():
    dikt_regions_admin     = {}
    csv_file = pd.read_csv(os.path.join(paths.extras,
                                        'corresp_poste_regionAdministrative.csv',
                                        ))
    for idx, (site, region) in csv_file.iterrows():
        name_site       = format_text.format_site_name(site)
        name_region     = (region.title()
                                 .replace('  ', ' ')
                                 .replace('-De-', '-de-')
                                 .replace(' De ', ' de ')
                                 .replace(' D\'', ' d\'')
                                 )
        dikt_regions_admin[name_site] = name_region
    return dikt_regions_admin


def get_dikt_regions_rte():
    dikt_regions_rte     = {}
    csv_file = pd.read_csv(os.path.join(paths.extras,
                                        'corresp_poste_regionRTE.csv',
                                        ))
    for idx, (site, region) in csv_file.iterrows():
        name_site       = format_text.format_site_name(site)
        name_region     = (region.replace('USE ', '' )
                                 .upper()
                                 )
        dikt_regions_rte[name_site] = name_region
    return dikt_regions_rte



