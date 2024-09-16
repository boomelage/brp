
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 14 02:27:39 2024

"""
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('term_structure')

from routine_ivol_collection import raw_ts
# raw_ts.drop_duplicates().to_csv(f"{generic} raw_ts.csv")


print('files processed')


