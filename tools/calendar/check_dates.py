#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy    as np
import datetime as dt
#




def check_dates(dates):
    assert (dates.hour.values == (np.arange(dates.shape[0]) % 24)).all()
    assert ((dates[24:] - dates[:-24]) == dt.timedelta(days = 1)).all()

