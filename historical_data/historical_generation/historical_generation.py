# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 13:54:06 2024

@author: boomelage
"""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
import pandas as pd
from data_query import dirdatacsv
os.chdir(current_dir)
