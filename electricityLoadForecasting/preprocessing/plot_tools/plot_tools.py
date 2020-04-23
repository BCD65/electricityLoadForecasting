
import os
import matplotlib.pyplot as plt
from termcolor import colored
from collections import OrderedDict
from pandas.plotting import register_matplotlib_converters; register_matplotlib_converters()
#

############################################################################### 

try:
    plt.rc('font', 
           **{'family': 'serif', 'serif': ['Computer Modern']}, 
           )
    plt.rc('text', 
           usetex=True, 
           )
    size_txt = 12
    plt.rc('legend',
           **{'fontsize':size_txt}, 
           )
    plt.rc('font',
           **{'size':size_txt}, 
           )
except Exception as e:
    print(colored(e, 'red'))

plt.ioff()
fig_size = (12,8)
lw       = 1
 
dikt_colors = { 
               'bad_win' : 'k', 
               'neg'     : 'g',
               'null'    : 'y',
               'too_low' : 'c', 
               'zero'    : 'r', 
               }

############################################################################### 
    
def plot_corrections(y, dates_unknown, y_hat_pred, path_plots_conso, regressor, rr_test, flags):
    fig, ax = plt.subplots(figsize = fig_size)
    ax.plot(y.index, 
            y, 
            linewidth = lw, 
            label     = y.name,
            )
    for ii, dd in enumerate(dates_unknown):
        plt.scatter(dd, y_hat_pred.loc[dd], marker = 'P', s = 20, color = dikt_colors[flags[dd]], zorder=10, label = flags[dd])
        plt.scatter(dd, 0,                  marker = '^', s = 20, color = dikt_colors[flags[dd]], zorder=10)
    handles, labels = ax.get_legend_handles_labels()
    labels   = [e.replace('_', ' ') for e in labels]
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    ax.set_xlabel('Date')
    ax.set_ylabel('Load')
    title =  '{0:6} - {1:6} corrections - score {2} test = {3:.4f}'.format(y.name,
                                                                          len(dates_unknown),
                                                                          regressor,
                                                                          rr_test,
                                                                          )
    title = title.replace('_', ' ')
    plt.title(title)
    fname = os.path.join(path_plots_conso, 
                         y.name.replace('_', ' '),
                         )
    os.makedirs(os.path.dirname(fname), exist_ok = True)
    fig.savefig(fname + '.png')
    plt.close(fig)

def plot_meteo(df_weather, path_plots_meteo):  
    print('plot_meteo')
    for ii, physical_quantity in enumerate(df_weather.columns.levels[1]):
        for jj, source in enumerate(df_weather.columns.levels[2]):
            meteo = df_weather.xs((physical_quantity, source), 
                                  level = [1,2],
                                  axis  = 1,
                                  )
            for kk, station in enumerate(meteo.columns):
                print('\r{0:2} / {1:2} - {2:2} / {3:2} - {4:2} / {5:2}'.format(ii,
                                                                               len(df_weather.columns.levels[1]),
                                                                               jj,
                                                                               len(df_weather.columns.levels[2]),
                                                                               kk,
                                                                               len(meteo.columns),
                                                                               ), end = '')
                fig, ax = plt.subplots(figsize = fig_size)
                ax.plot(meteo.index, 
                        meteo[station], 
                        linewidth = lw,
                        )
                ax.set_xlabel('Date')
                title = ' - '.join([station, physical_quantity, source])
                title = title.replace('_', ' ')
                plt.title(title)
                plt.tight_layout()
                fname = os.path.join(path_plots_meteo, source, physical_quantity, title)
                os.makedirs(os.path.dirname(fname), exist_ok = True)
                fig.savefig(fname + '.png')
                plt.close(fig)
    print('\nFinished plot_meteo')


def plot_trash(trash, load_data, path_plots_trash): 
    print('plot_trash')
    for ii, (substation, reason) in enumerate(trash):
        fig, ax = plt.subplots(figsize = fig_size)
        ax.plot(load_data.index, 
                load_data[substation], 
                linewidth = lw,
                )
        title = '{0} - trashed : {1}'.format(substation, 
                                             reason, 
                                             )
        title = title.replace('_', ' ')
        plt.title(title)
        plt.tight_layout()
        fname = os.path.join(path_plots_trash, 
                             substation.replace('.', '_'),
                             )
        os.makedirs(os.path.dirname(fname), exist_ok = True)
        fig.savefig(fname + '.png')
        plt.close(fig)
        print('{0:5}/{1:5} {2:6} trashed'.format(ii, len(trash), substation))
    print('Finished plot_trash')
    


