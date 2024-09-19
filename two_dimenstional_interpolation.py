#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 19 14:17:18 2024

@author: doomd
"""

import os
import sys
import time
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.append('term_structure')
sys.path.append('contract_details')
sys.path.append('misc')

import QuantLib as ql
import numpy as np

import pandas as pd

from derman_test import atm_volvec, derman_coefs
from routine_calibration_generation import T, call_ks, put_ks


ks = np.array((call_ks, put_ks), dtype=float)
ks = np.unique(ks.flatten())

atm_volvec = atm_volvec.loc[T]


X = T
Y = ks

Z = [[(x-3)**2 + y for x in X] for y in Y]
df = pd.DataFrame(Z, columns=X, index=Y)


print(df)
# i = ql.BilinearInterpolation(X, Y, Z)

# XX = np.linspace(0, 5, 9)
# YY = np.linspace(0.55, 1.0, 10)

# extrapolated = pd.DataFrame(
#     [[i(x,y, True) for x in XX] for y in YY],
#     columns=XX,
#     index=YY)





# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm

# fig = plt.figure()
# ax = fig.gca()#projection='3d'
# ax.set_title("Surface Interpolation")

# Xs, Ys = np.meshgrid(XX, YY)
# surf = ax.plot_surface(
#     Xs, Ys, extrapolated, rstride=1, cstride=1, cmap=cm.coolwarm
# )
# fig.colorbar(surf, shrink=0.5, aspect=5);