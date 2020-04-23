
import pandas as pd
#




def check_dates(dates):
    assert ((dates[1:]  - dates[:-1] ) == pd.Timedelta(hours = 1)).all()
    assert ((dates[24:] - dates[:-24]) == pd.Timedelta(days  = 1)).all()

