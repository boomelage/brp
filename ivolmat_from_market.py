#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 19:25:46 2024

"""

import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import QuantLib as ql



