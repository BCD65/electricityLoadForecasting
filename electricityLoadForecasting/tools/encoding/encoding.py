#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy    as np
import datetime as dt
import chardet
#


def find_encoding(fname):
    r_file = open(fname, 'rb').read()
    result = chardet.detect(r_file)
    charenc = result['encoding']
    return charenc

