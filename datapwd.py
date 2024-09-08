#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 12:00:05 2024
"""

import os

def pwd():
    pwd = str(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(pwd)
    filenames = os.listdir(pwd)
    for name in filenames:
        if name.endswith(('.xlsx','.csv','.xls')):
            print(name)
        else:
            pass