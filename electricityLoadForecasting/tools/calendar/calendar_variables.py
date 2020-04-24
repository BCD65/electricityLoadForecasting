


import numpy as np
import pandas as pd
import datetime as dt
import astral
#
from .calendar_infos import EasterMondays, Ascensions, Pentecosts, smr_intervals, xmas_intervals


def binary_christmas(date):
    for db, de in xmas_intervals:
        assert db.weekday() == 5
        assert de.weekday() == 6
    assert date.date() > xmas_intervals[0][0]
    assert date.date() < xmas_intervals[-1][1]
    for db, de in xmas_intervals:
        if date.date() >= db and date.date() <= de:
            return 1
    return 0


def binary_holidays(date):    
    day   = date.day
    month = date.month
    year  = date.year    
    date_no_time = dt.date(day = day, month = month, year = year)
    if day == 1 and month == 1 : 
        return 1 #1 Janvier    
    if day == 1 and month == 5 : 
        return 1 #1 Mai    
    if day == 8 and month == 5 : 
        return 1 #8 Mai    
    if day == 14 and month == 7 : 
        return 1 #14 Juilland    
    if day == 15 and month == 8 : 
        return 1 #15 Août    
    if day == 1 and month == 11 : 
        return 1 #1 Novembre    
    if day == 11 and month == 11 : 
        return 1 #11 Novembre    
    if day == 25 and month == 12 : 
        return 1 #25 Décembre    
    if date_no_time in EasterMondays:
        return 1    
    if date_no_time in Ascensions: 
        return 1    
    if date_no_time in Pentecosts:
        return 1    
    return 0


def binary_summer_break(date):
    for db, de in smr_intervals:
        assert db.weekday() == 5
        assert de.weekday() == 6
    assert date.date() > smr_intervals[0][0]
    assert date.date() < smr_intervals[-1][1]
    for db, de in smr_intervals:
        if date.date() >= db and date.date() <= de:
            return 1
    return 0


def binary_daytime(samples_dt):
    city_name          = 'Paris'
    a                  = astral.Astral()
    a.solar_depression = 'civil'
    city               = a[city_name]
    sun          = pd.DataFrame(list(map(lambda dd : city.sun(date = dd, local = False), samples_dt.date))) # This takes a lot of time !!!
    daylights    = np.logical_and(samples_dt > sun['sunrise'],
                                  samples_dt < sun['sunset'],
                                  )
    return daylights
 
    
def compute_calendar_variables(samples_dt):
    calendar_variables = pd.DataFrame({
                                       'year_day'      :  samples_dt.dayofyear,
                                       'week_day'      :  samples_dt.dayofweek,
                                       'weekend'       : (samples_dt.dayofweek > 4).astype(int),
                                       'week_hour'     :  samples_dt.hour + 24 * samples_dt.dayofweek,
                                       'hour'          :  samples_dt.hour,
                                       'timestamp'     : list(map(lambda dd : dd.timestamp(), samples_dt)),                                       
                                       'holidays'      : list(map(binary_holidays,     samples_dt)),
                                       'summer_break'  : list(map(binary_summer_break, samples_dt)),
                                       'xmas'          : list(map(binary_christmas,    samples_dt)),
                                       'daylight'      : binary_daytime(samples_dt),
                                       }, 
                                      index = samples_dt,
                                      )  
    return calendar_variables 
