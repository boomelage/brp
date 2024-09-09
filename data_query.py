#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 12:00:05 2024
"""
import os
def dirdata():
    pwd = os.path.dirname(os.path.abspath(__file__))
    os.chdir(pwd)
    filenames = os.listdir(pwd)
    data_files = []
    for name in filenames:
        if name.endswith('.xlsx'): 
            data_files.append(name)
    return data_files  