
import os
#


### Dataset ###
LEVEL = 'France' # 'France' # 'administrative_regions' 
YEARS_WEATHER = range(2010, 2020)
YEARS_LOADS   = range(2013, 2019)

### Paths ###
root_folder           = os.path.join(os.path.expanduser("~"),
                                     'Eco2mix_forecasting',
                                     )

### Activation plots ###
bool_plot = True
bool_plot_corrections = True and bool_plot
bool_plot_trash       = True and bool_plot
bool_plot_meteo       = True and bool_plot



