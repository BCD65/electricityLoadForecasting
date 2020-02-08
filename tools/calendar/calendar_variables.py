


import numpy as np
import pandas as pd
import datetime as dt
import astral
#
from .calendar_infos import EasterMondays, Ascensions, Pentecosts, smr_intervals


def binary_christmas(date):
    xmas_intervals = [
                      (dt.date(year = 2012, month = 12, day = 22), 
                       dt.date(year = 2013, month =  1, day =  6),
                       ),
                      (dt.date(year = 2013, month = 12, day = 21), 
                       dt.date(year = 2014, month =  1, day =  5),
                       ),
                      (dt.date(year = 2014, month = 12, day = 20), 
                       dt.date(year = 2015, month =  1, day =  4),
                       ),
                      (dt.date(year = 2015, month = 12, day = 19), 
                       dt.date(year = 2016, month =  1, day =  3),
                       ),
                      (dt.date(year = 2016, month = 12, day = 17),
                       dt.date(year = 2017, month =  1, day =  1),
                       ),
                      (dt.date(year = 2017, month = 12, day = 23), 
                       dt.date(year = 2018, month =  1, day =  7),
                       ),
                      ]
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


def binary_daytime(dates):
    city_name          = 'Paris'
    a                  = astral.Astral()
    a.solar_depression = 'civil'
    city               = a[city_name]
    daylights = np.zeros(dates.shape)
    for ii, dd in enumerate(dates):
        sun           = city.sun(date=dd.date(), local = False)
        daylights[ii] = (dd>sun['sunrise'])*(dd<sun['sunset'])
    return daylights
 

def compute_calendar_variables(dates):
    calendar_variables = pd.DataFrame({
                                       'year_day'      : dates.dayofyear,
                                       'week_day'      : dates.dayofweek,
                                       'weekend'       : dates.dayofweek > 4,
                                       'week_hour'     : dates.hour + 24 * dates.dayofweek,
                                       'hour'          : dates.hour,
                                       'timestamp'     : list(map(lambda dd : dd.timestamp(), dates)),                                       
                                       'holidays'      : list(map(binary_holidays,     dates)),
                                       'summer_break'  : list(map(binary_summer_break, dates)),
                                       'xmas'          : list(map(binary_christmas,    dates)),
                                       'daylight'      : binary_daytime(dates),
                                       }, 
                                      index = dates,
                                      )      
    return calendar_variables 
