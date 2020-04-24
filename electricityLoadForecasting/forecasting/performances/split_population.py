
import pandas as pd
#


def split_population(hprm, freq = None):
    if hprm['training_set.form'] == 'continuous':
        training_dates   = pd.date_range(start  = hprm['training_set.first_sample'],
                                         end    = hprm['training_set.first_sample'] + hprm['training_set.length'],
                                         freq   = freq,
                                         closed = 'left',
                                         )
        validation_dates = pd.date_range(start  = training_dates[-1] + training_dates.freq,
                                         end    = training_dates[-1] + training_dates.freq + hprm['validation_set.length'],
                                         freq   = freq,
                                         closed = 'left',
                                         )
    else:
        # For discontinuous training set
        raise NotImplementedError
    return training_dates, validation_dates