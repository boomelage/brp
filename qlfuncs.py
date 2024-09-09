# -*- coding: utf-8 -*-
"""
Created on Mon Sep  9 13:54:57 2024

@author: boomelage
"""

import QuantLib as ql

def make_ql_array(size,nparr):
    qlarr = ql.Array(size,1)
    for i in range(size):
        qlarr[i] = float(nparr[i])
    return qlarr
    
    

