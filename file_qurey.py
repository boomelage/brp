#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 12:00:05 2024

"""

import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)
filenames = os.listdir(pwd)
for i in range(len(filenames)):
    if filenames[i].endswith('.xlsx'):
        print(f"{i+1} {filenames[i]}")
    else:
        pass