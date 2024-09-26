#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 12:00:05 2024
"""
import os

def dirdata(exclude_file=None):
    filenames = os.listdir()
    data_files = []
    for name in filenames:
        if name.endswith('.xlsx') and name != exclude_file: 
            data_files.append(name)
    return data_files

def dirdatacsv(exclude_file=None):
    filenames = os.listdir()
    data_files = []
    for name in filenames:
        if name.endswith('.csv') and name != exclude_file: 
            data_files.append(name)
    return data_files

