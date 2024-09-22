#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 20:34:01 2024

@author: doomd
"""
import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
from settings import model_settings
ms = model_settings()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from settings import model_settings
ms = model_settings()
import time
from datetime import datetime
from PIL import Image
from tqdm import tqdm
import QuantLib as ql
from bicubic_interpolation import K, make_bicubic_ts
os.chdir(current_dir)


"""
generate frames

"""

T = np.arange(1,360,1).astype(float)
K = np.linspace(min(K),max(K),5000)
bicubic_ts = make_bicubic_ts(T,K)

ql_bicubic = ql.Matrix(len(bicubic_ts.index),len(bicubic_ts.columns),0.00)

print("space generated")

for i, k in enumerate(K):
    for j, t in enumerate(T):
        ql_bicubic[i][j] = bicubic_ts.loc[k,t]
        
expiration_dates = ms.compute_ql_maturity_dates(T)
black_var_surface = ms.make_black_var_surface(expiration_dates, K, ql_bicubic)
    

plt.rcParams['figure.figsize'] = (10, 10)
plot_maturities = np.sort(np.array(T,dtype=float)/365)
plot_strikes = np.sort(K).astype(float)
X, Y = np.meshgrid(plot_strikes, plot_maturities)
Z = np.array([[
    black_var_surface.blackVol(y, x) for x in plot_strikes] 
    for y in plot_maturities])
azims = np.arange(0,360,1)
print("object generated")


for i, azim in enumerate(azims):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.view_init(elev=30, azim=azim)  
    ax.set_title('bicubic interpolation of volatility surface approximated via Derman')
    surf = ax.plot_surface(X,Y,Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                    linewidth=0.1)
    fig.colorbar(surf, shrink=0.5, aspect=5)
    ax.set_xlabel("Strikes", size=9)
    ax.set_ylabel("Maturities (Years)", size=9)
    ax.set_zlabel("Volatility", size=9)
    
    plt.show()
    fig.savefig(f'{int(azim+1)}.png')
    plt.close(fig)
    print(f"{int(i+1)}/{len(azims)}")


def create_stop_motion_gif_from_directory(output_path, durations, loop=0):
    # Get all PNG files in the current working directory
    image_paths = [f for f in os.listdir() if f.endswith('.png')]
    image_paths = sorted(image_paths, key=lambda x: int(x.split('.')[0]))
    
    if not image_paths:
        print("No PNG files found in the current directory.")
        return

    # Open images from paths
    images = [Image.open(image) for image in image_paths]
    
    # Save the images as a GIF
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=durations,  # Duration of each frame in milliseconds
        loop=loop  # 0 for infinite loop, or set the number of loops
    )
    print(f"GIF saved as {output_path}")

gif_time = time.time()
gif_dtime = datetime.fromtimestamp(gif_time)
gif_tag = gif_dtime.strftime("%Y-%m-%d %H-%M-%S")
create_stop_motion_gif_from_directory(f'volsurf {gif_tag}.gif', durations=0.015, loop=0)

