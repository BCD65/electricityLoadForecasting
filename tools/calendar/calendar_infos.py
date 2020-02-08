
import datetime as dt


EasterMondays = [
          dt.date(day = 5,  month = 4, year = 2010), 
          dt.date(day = 25, month = 4, year = 2011), 
          dt.date(day = 9,  month = 4, year = 2012), 
          dt.date(day = 1,  month = 4, year = 2013), 
          dt.date(day = 21, month = 4, year = 2014), 
          dt.date(day = 6,  month = 4, year = 2015), 
          dt.date(day = 28, month = 3, year = 2016), 
          dt.date(day = 27, month = 3, year = 2017), 
          dt.date(day = 2,  month = 4, year = 2018), 
          dt.date(day = 22, month = 4, year = 2019), 
          ]
    
Ascensions = list(map(lambda x : x + dt.timedelta(days = 38), EasterMondays))
Pentecosts = list(map(lambda x : x + dt.timedelta(days = 49), EasterMondays))

for date in EasterMondays:
    assert date.weekday() == 0, (date.weekday(), date, 'Eastermonday')
for date in Ascensions:
    assert date.weekday() == 3, (date.weekday(), date, 'Ascension')
for date in Pentecosts:
    assert date.weekday() == 0, (date.weekday(), date, 'Pentecost')
    
    
smr_intervals = [
                 (dt.date(year = 2012, month = 7, day = 14), 
                  dt.date(year = 2012, month = 8, day = 26),
                  ),
                 (dt.date(year = 2013, month = 7, day = 13), 
                  dt.date(year = 2013, month = 8, day = 25),
                  ),
                 (dt.date(year = 2014, month = 7, day = 19), 
                  dt.date(year = 2014, month = 8, day = 31),
                  ),
                 (dt.date(year = 2015, month = 7, day = 18), 
                  dt.date(year = 2015, month = 8, day = 30),
                  ),
                 (dt.date(year = 2016, month = 7, day = 16), 
                  dt.date(year = 2016, month = 8, day = 28),
                  ),
                 (dt.date(year = 2017, month = 7, day = 15), 
                  dt.date(year = 2017, month = 8, day = 27),
                  ),
                 (dt.date(year = 2018, month = 7, day = 14), 
                  dt.date(year = 2018, month = 8, day = 26),
                  ),
                 ]