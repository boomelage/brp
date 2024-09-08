#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
pwd = str(os.path.dirname(os.path.abspath(__file__)))
os.chdir(pwd)
"""
Created on Sun Sep  8 03:09:26 2024

"""

from bloomberg_ivols import generate_from_market_data    

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(15,7)
plt.style.use("dark_background")
from matplotlib import cm

dataset, ivol_table, implied_vols_matrix, black_var_surface, strikes,\
    maturities = generate_from_market_data(0.00, 0.00)



strikes_grid = strikes
expiry = 0.3643 # years
implied_vols = [black_var_surface.blackVol(expiry, s)
                for s in strikes_grid] # can interpolate here
actual_data = ivol_table[4]
fig, ax = plt.subplots()
ax.plot(strikes_grid, implied_vols, label="Black Surface")
ax.plot(strikes, actual_data, "o", label="Actual")
ax.set_xlabel("Strikes", size=12)
ax.set_ylabel("Vols", size=12)
legend = ax.legend(loc="upper right")

plottmin = (min(maturities)+1)/365
plottmax = (max(maturities)-1)/365


plot_maturities = np.linspace(plottmin, plottmax, len(maturities))

print(f"printing plot maturities: {plot_maturities}")
# Generate the mesh grid
X, Y = np.meshgrid(strikes, plot_maturities)

# Query the implied volatilities for each (strike, maturity) pair
Z = np.array([[black_var_surface.blackVol(y, x) for x in strikes] for y in maturities])

# Plot the 3D surface
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.1)
fig.colorbar(surf, shrink=0.5, aspect=5)

ax.set_xlabel("Strikes", size=12)
ax.set_ylabel("Maturities (Years)", size=12)
ax.set_zlabel("Volatility", size=12)

plt.show()


