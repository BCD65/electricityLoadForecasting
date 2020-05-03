#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run to prepare the raw data (main function is at the end)
It detects corrupted data
It proposes modifications of the corrupted data
It saves everything in the end
This script takes about 24 hours and has not been optimized but only has to be run once.
"""


from electricityLoadForecasting.preprocessing import eCO2mix


if __name__ == '__main__':
    eCO2mix.main()
        