#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  8 12:00:05 2024
"""
import os
def dirdata():
    pwd = os.path.dirname(os.path.abspath(__file__))  # get current directory of the script
    os.chdir(pwd)  # change to this directory
    filenames = os.listdir(pwd)  # list all files in the directory
    data_files = []
    
    for name in filenames:
        if name.endswith('.xlsx'):  # check if file ends with .xlsx
            data_files.append(name)  # append to data_files list
    
    for data_file in data_files:
        print(data_file)  # print all .xlsx files

    return data_files  # return the list of .xlsx files

# Example usage
if __name__ == "__main__":
    data_files = dirdata()