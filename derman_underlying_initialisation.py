#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 15:23:06 2024

@author: doomd
"""

from Derman import derman

derman = derman()


ks, mats, ts = derman.retrieve_ts()

derman_coefs = derman.get_derman_coefs()
