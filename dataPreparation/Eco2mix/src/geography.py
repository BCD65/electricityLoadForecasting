
import pandas as pd
import electricityLoadForecasting


coordinates_sites = {'France'                 : pd.DataFrame(data    = [[46,2]], 
                                                             index   = pd.Index(['France'], 
                                                                                name = electricityLoadForecasting.src.user_site_name,
                                                                                ),
                                                             columns = [electricityLoadForecasting.src.user_latitude, 
                                                                        electricityLoadForecasting.src.user_longitude,
                                                                        ],
                                                             ),
                     'administrative_regions' : pd.DataFrame(data    = [[45.5, 4.6],
                                                                        [47.1, 4.8],
                                                                        [48.1, -2.8],
                                                                        [47.5, 1.8],
                                                                        [48.7, 5.8],
                                                                        [49.9, 2.9],
                                                                        [48.8, 2.5],
                                                                        [49.0, 0.3],
                                                                        [45.2, 0.0],
                                                                        [43.6, 2.3],
                                                                        [43.8, 6.4],
                                                                        [47.5, -0.7],
                                                                        ],# manually selected
                                                             index   = pd.Index(['Auvergne-Rhône-Alpes',
                                                                                 'Bourgogne-Franche-Comté',
                                                                                 'Bretagne',
                                                                                 'Centre-Val-de-Loire',
                                                                                 'Grand-Est', 
                                                                                 'Hauts-de-France',
                                                                                 'Ile-de-France',
                                                                                 'Normandie',
                                                                                 'Nouvelle-Aquitaine',
                                                                                 'Occitanie',
                                                                                 'PACA',
                                                                                 'Pays-de-la-Loire',
                                                                                 ],
                                                                                name = electricityLoadForecasting.src.user_site_name,
                                                                                ),
                                                                        columns = [electricityLoadForecasting.src.user_latitude,
                                                                                   electricityLoadForecasting.src.user_longitude,
                                                                                   ],
                                                             ),
                     }


