```python
import numpy as np
from scipy import interpolate
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import QuantLib as ql

#Let's create some random  data
array = np.random.randint(0,10,(10,10)).astype(float)
#values grater then 7 goes to np.nan
array[array>7] = np.nan
```


```python
array
```




    array([[ 3.,  0.,  4.,  4., nan,  7.,  2.,  3., nan,  7.],
           [ 2.,  5.,  2.,  2., nan, nan,  6.,  5.,  0.,  2.],
           [ 7.,  3.,  4.,  0.,  3.,  1.,  3.,  2.,  7.,  7.],
           [ 7.,  6.,  6.,  5.,  7., nan,  7.,  4., nan,  2.],
           [ 4., nan, nan,  1.,  6.,  7.,  4.,  0., nan,  6.],
           [ 3.,  0.,  0.,  0.,  0.,  7.,  4.,  4., nan,  3.],
           [ 2.,  4., nan,  2.,  5.,  2.,  6.,  0.,  5., nan],
           [nan, nan,  4.,  1., nan,  2.,  0.,  7.,  1.,  7.],
           [ 3.,  5., nan,  1.,  6.,  3.,  4.,  5.,  5., nan],
           [nan, nan, nan,  0.,  1.,  2., nan,  0.,  0.,  7.]])




```python
x = np.arange(0, array.shape[1])
y = np.arange(0, array.shape[0])
#mask invalid values
array = np.ma.masked_invalid(array)
xx, yy = np.meshgrid(x, y)
#get only the valid values
x1 = xx[~array.mask]
y1 = yy[~array.mask]
newarr = array[~array.mask]

GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                          (xx, yy),
                             method='cubic')
```


```python
GD1
```




    array([[3.        , 0.        , 4.        , 4.        , 5.99868799,
            7.        , 2.        , 3.        , 3.97263616, 7.        ],
           [2.        , 5.        , 2.        , 2.        , 3.11250958,
            2.35281167, 6.        , 5.        , 0.        , 2.        ],
           [7.        , 3.        , 4.        , 0.        , 3.        ,
            1.        , 3.        , 2.        , 7.        , 7.        ],
           [7.        , 6.        , 6.        , 5.        , 7.        ,
            4.8529526 , 7.        , 4.        , 3.81893587, 2.        ],
           [4.        , 3.24100122, 2.2032387 , 1.        , 6.        ,
            7.        , 4.        , 0.        , 1.329631  , 6.        ],
           [3.        , 0.        , 0.        , 0.        , 0.        ,
            7.        , 4.        , 4.        , 5.09273677, 3.        ],
           [2.        , 4.        , 2.19884159, 2.        , 5.        ,
            2.        , 6.        , 0.        , 5.        , 4.18079331],
           [2.72978172, 5.73410145, 4.        , 1.        , 6.6036695 ,
            2.        , 0.        , 7.        , 1.        , 7.        ],
           [3.        , 5.        , 2.27498131, 1.        , 6.        ,
            3.        , 4.        , 5.        , 5.        , 8.02238575],
           [       nan,        nan,        nan, 0.        , 1.        ,
            2.        , 1.36059425, 0.        , 0.        , 7.        ]])




```python
from historical_alphaVantage_collection import ivol_df
ivol_df
```

    
    pricing settings:
    Actual/365 (Fixed) day counter
    New York stock exchange calendar
    compounding: continuous
    frequency: annual
    
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.0</th>
      <th>1.0</th>
      <th>2.0</th>
      <th>5.0</th>
      <th>6.0</th>
      <th>7.0</th>
      <th>8.0</th>
      <th>9.0</th>
      <th>13.0</th>
      <th>14.0</th>
      <th>16.0</th>
      <th>23.0</th>
      <th>28.0</th>
      <th>30.0</th>
      <th>37.0</th>
      <th>44.0</th>
      <th>57.0</th>
      <th>72.0</th>
      <th>85.0</th>
      <th>107.0</th>
      <th>118.0</th>
      <th>149.0</th>
      <th>170.0</th>
      <th>177.0</th>
      <th>261.0</th>
      <th>271.0</th>
      <th>352.0</th>
      <th>363.0</th>
      <th>380.0</th>
      <th>443.0</th>
      <th>534.0</th>
      <th>716.0</th>
      <th>744.0</th>
      <th>1080.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>120.0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.03383</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>130.0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.03286</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>140.0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.03189</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>150.0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.04166</td>
      <td>NaN</td>
      <td>0.03092</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.02040</td>
      <td>0.01522</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>155.0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.04100</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.02008</td>
      <td>0.01498</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>715.0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.13050</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.08828</td>
      <td>0.07075</td>
      <td>0.06861</td>
      <td>0.10443</td>
    </tr>
    <tr>
      <th>720.0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.61982</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.13263</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.10657</td>
    </tr>
    <tr>
      <th>730.0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.11343</td>
    </tr>
    <tr>
      <th>740.0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.11007</td>
    </tr>
    <tr>
      <th>750.0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.10855</td>
    </tr>
  </tbody>
</table>
<p>295 rows × 34 columns</p>
</div>




```python
ivol_df = ivol_df.dropna(how='all',axis=0).dropna(how='all',axis=1)
print(f"\nT: \n{ivol_df.columns.tolist()}\n")
print(f"\nK: \n{ivol_df.index.tolist()}\n")
```

    
    T: 
    [0.0, 1.0, 2.0, 5.0, 6.0, 7.0, 8.0, 9.0, 13.0, 14.0, 16.0, 23.0, 28.0, 30.0, 37.0, 44.0, 57.0, 72.0, 85.0, 107.0, 118.0, 149.0, 170.0, 177.0, 261.0, 271.0, 352.0, 363.0, 380.0, 443.0, 534.0, 716.0, 744.0, 1080.0]
    
    
    K: 
    [120.0, 130.0, 140.0, 150.0, 155.0, 160.0, 165.0, 170.0, 175.0, 180.0, 185.0, 190.0, 195.0, 200.0, 205.0, 210.0, 215.0, 220.0, 225.0, 230.0, 235.0, 240.0, 245.0, 250.0, 255.0, 260.0, 265.0, 270.0, 275.0, 280.0, 285.0, 290.0, 295.0, 300.0, 305.0, 310.0, 315.0, 320.0, 325.0, 326.0, 327.0, 328.0, 329.0, 330.0, 331.0, 332.0, 333.0, 334.0, 335.0, 336.0, 337.0, 338.0, 339.0, 340.0, 341.0, 342.0, 343.0, 344.0, 345.0, 346.0, 347.0, 348.0, 349.0, 350.0, 351.0, 352.0, 353.0, 354.0, 355.0, 356.0, 357.0, 358.0, 359.0, 360.0, 361.0, 362.0, 363.0, 364.0, 365.0, 366.0, 367.0, 368.0, 369.0, 370.0, 371.0, 372.0, 373.0, 374.0, 375.0, 376.0, 377.0, 378.0, 379.0, 380.0, 381.0, 382.0, 383.0, 384.0, 385.0, 386.0, 387.0, 388.0, 389.0, 390.0, 391.0, 392.0, 393.0, 394.0, 395.0, 396.0, 397.0, 398.0, 399.0, 400.0, 401.0, 402.0, 403.0, 404.0, 405.0, 406.0, 407.0, 408.0, 409.0, 410.0, 411.0, 412.0, 413.0, 414.0, 415.0, 416.0, 417.0, 418.0, 419.0, 420.0, 421.0, 422.0, 423.0, 424.0, 425.0, 426.0, 427.0, 428.0, 429.0, 430.0, 431.0, 432.0, 433.0, 434.0, 435.0, 436.0, 437.0, 438.0, 439.0, 440.0, 441.0, 442.0, 442.5, 443.0, 444.0, 445.0, 446.0, 447.0, 447.5, 448.0, 449.0, 450.0, 451.0, 452.0, 452.5, 453.0, 454.0, 455.0, 456.0, 457.0, 457.5, 458.0, 459.0, 460.0, 461.0, 462.0, 462.5, 463.0, 464.0, 465.0, 466.0, 467.0, 467.5, 468.0, 469.0, 470.0, 471.0, 472.0, 472.5, 473.0, 474.0, 475.0, 476.0, 477.0, 477.5, 478.0, 479.0, 480.0, 481.0, 482.0, 482.5, 483.0, 484.0, 485.0, 486.0, 487.0, 487.5, 488.0, 489.0, 490.0, 491.0, 492.0, 493.0, 494.0, 495.0, 496.0, 497.0, 498.0, 499.0, 500.0, 501.0, 502.0, 503.0, 504.0, 505.0, 506.0, 507.0, 508.0, 509.0, 510.0, 511.0, 512.0, 513.0, 514.0, 515.0, 516.0, 517.0, 518.0, 519.0, 520.0, 521.0, 522.0, 523.0, 524.0, 525.0, 526.0, 527.0, 528.0, 529.0, 530.0, 535.0, 540.0, 545.0, 550.0, 555.0, 560.0, 565.0, 570.0, 575.0, 580.0, 585.0, 590.0, 595.0, 600.0, 605.0, 610.0, 615.0, 620.0, 625.0, 630.0, 635.0, 640.0, 645.0, 650.0, 655.0, 660.0, 665.0, 670.0, 675.0, 680.0, 685.0, 690.0, 695.0, 700.0, 705.0, 710.0, 715.0, 720.0, 730.0, 740.0, 750.0]
    
    


```python
strikes = ivol_df.iloc[:,0].dropna().index
ivol_df = ivol_df.loc[strikes,:].copy()
T = ivol_df.columns.tolist()
K = ivol_df.index.tolist()
ivol_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.0</th>
      <th>1.0</th>
      <th>2.0</th>
      <th>5.0</th>
      <th>6.0</th>
      <th>7.0</th>
      <th>8.0</th>
      <th>9.0</th>
      <th>13.0</th>
      <th>14.0</th>
      <th>16.0</th>
      <th>23.0</th>
      <th>28.0</th>
      <th>30.0</th>
      <th>37.0</th>
      <th>44.0</th>
      <th>57.0</th>
      <th>72.0</th>
      <th>85.0</th>
      <th>107.0</th>
      <th>118.0</th>
      <th>149.0</th>
      <th>170.0</th>
      <th>177.0</th>
      <th>261.0</th>
      <th>271.0</th>
      <th>352.0</th>
      <th>363.0</th>
      <th>380.0</th>
      <th>443.0</th>
      <th>534.0</th>
      <th>716.0</th>
      <th>744.0</th>
      <th>1080.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>402.0</th>
      <td>1.14335</td>
      <td>1.19435</td>
      <td>0.84740</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.36052</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.27272</td>
      <td>NaN</td>
      <td>0.24284</td>
      <td>0.20458</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.18309</td>
      <td>NaN</td>
      <td>0.16449</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>403.0</th>
      <td>1.12623</td>
      <td>1.17682</td>
      <td>0.83475</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.35748</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.27013</td>
      <td>NaN</td>
      <td>0.24071</td>
      <td>0.20382</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.18248</td>
      <td>NaN</td>
      <td>0.16465</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>404.0</th>
      <td>1.10911</td>
      <td>1.15929</td>
      <td>0.82210</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.35199</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.26769</td>
      <td>NaN</td>
      <td>0.23858</td>
      <td>0.20306</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.18218</td>
      <td>NaN</td>
      <td>0.16465</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>405.0</th>
      <td>1.09199</td>
      <td>1.14176</td>
      <td>0.82408</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.45092</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.34864</td>
      <td>0.30702</td>
      <td>0.27150</td>
      <td>0.28568</td>
      <td>0.27044</td>
      <td>0.26510</td>
      <td>0.24803</td>
      <td>0.23720</td>
      <td>0.20138</td>
      <td>0.18782</td>
      <td>0.18599</td>
      <td>0.19300</td>
      <td>0.18675</td>
      <td>0.18157</td>
      <td>0.16968</td>
      <td>0.16449</td>
      <td>0.16068</td>
      <td>0.15398</td>
      <td>0.14681</td>
      <td>0.14742</td>
      <td>0.13462</td>
      <td>0.09727</td>
      <td>0.00293</td>
      <td>0.00202</td>
    </tr>
    <tr>
      <th>406.0</th>
      <td>1.07487</td>
      <td>1.12438</td>
      <td>0.81143</td>
      <td>0.35829</td>
      <td>0.30711</td>
      <td>0.26872</td>
      <td>0.46327</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.34315</td>
      <td>NaN</td>
      <td>0.26845</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.26251</td>
      <td>0.24605</td>
      <td>0.23538</td>
      <td>0.20047</td>
      <td>NaN</td>
      <td>0.18507</td>
      <td>0.19239</td>
      <td>NaN</td>
      <td>0.18111</td>
      <td>NaN</td>
      <td>0.16434</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>525.0</th>
      <td>0.74192</td>
      <td>0.74192</td>
      <td>0.52394</td>
      <td>0.33019</td>
      <td>0.30108</td>
      <td>0.27851</td>
      <td>0.26022</td>
      <td>0.24498</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.18218</td>
      <td>0.15077</td>
      <td>0.13584</td>
      <td>0.13080</td>
      <td>0.11678</td>
      <td>0.11846</td>
      <td>0.11084</td>
      <td>0.10611</td>
      <td>0.10367</td>
      <td>0.10199</td>
      <td>0.10199</td>
      <td>0.10504</td>
      <td>0.10581</td>
      <td>0.10535</td>
      <td>0.11023</td>
      <td>0.10977</td>
      <td>0.11770</td>
      <td>0.11663</td>
      <td>0.11648</td>
      <td>0.12029</td>
      <td>0.12166</td>
      <td>0.12105</td>
      <td>0.11968</td>
      <td>0.11419</td>
    </tr>
    <tr>
      <th>530.0</th>
      <td>0.79801</td>
      <td>0.79801</td>
      <td>0.56372</td>
      <td>0.35534</td>
      <td>0.32409</td>
      <td>0.29970</td>
      <td>0.28004</td>
      <td>0.26373</td>
      <td>NaN</td>
      <td>0.21022</td>
      <td>0.19620</td>
      <td>0.16251</td>
      <td>0.14635</td>
      <td>0.14117</td>
      <td>0.12608</td>
      <td>0.12776</td>
      <td>0.11419</td>
      <td>0.11007</td>
      <td>0.10596</td>
      <td>0.10367</td>
      <td>0.10291</td>
      <td>0.10459</td>
      <td>0.10474</td>
      <td>0.10443</td>
      <td>0.10901</td>
      <td>0.10809</td>
      <td>0.11587</td>
      <td>0.11480</td>
      <td>0.11617</td>
      <td>0.11861</td>
      <td>0.12029</td>
      <td>0.12013</td>
      <td>0.11526</td>
      <td>0.11449</td>
    </tr>
    <tr>
      <th>535.0</th>
      <td>0.85320</td>
      <td>0.85320</td>
      <td>0.60259</td>
      <td>0.38004</td>
      <td>0.34650</td>
      <td>0.32059</td>
      <td>0.29955</td>
      <td>0.28217</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.21007</td>
      <td>0.17394</td>
      <td>0.15687</td>
      <td>0.15123</td>
      <td>0.13523</td>
      <td>0.12303</td>
      <td>0.12227</td>
      <td>0.11434</td>
      <td>0.10992</td>
      <td>0.10504</td>
      <td>0.10398</td>
      <td>0.10428</td>
      <td>0.10413</td>
      <td>0.10352</td>
      <td>0.10763</td>
      <td>0.10703</td>
      <td>0.11434</td>
      <td>0.11327</td>
      <td>0.11480</td>
      <td>0.11693</td>
      <td>0.11876</td>
      <td>0.11892</td>
      <td>0.11785</td>
      <td>0.11434</td>
    </tr>
    <tr>
      <th>540.0</th>
      <td>0.90746</td>
      <td>0.90746</td>
      <td>0.64101</td>
      <td>0.40427</td>
      <td>0.36876</td>
      <td>0.34101</td>
      <td>0.31876</td>
      <td>0.30016</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.22364</td>
      <td>0.18538</td>
      <td>0.16708</td>
      <td>0.16114</td>
      <td>0.14422</td>
      <td>0.13126</td>
      <td>0.13050</td>
      <td>0.11968</td>
      <td>0.11267</td>
      <td>0.10763</td>
      <td>0.10703</td>
      <td>0.10428</td>
      <td>0.10428</td>
      <td>0.10337</td>
      <td>0.10642</td>
      <td>0.10581</td>
      <td>0.11297</td>
      <td>0.11206</td>
      <td>0.11190</td>
      <td>0.11556</td>
      <td>0.11739</td>
      <td>0.11800</td>
      <td>0.11693</td>
      <td>0.11404</td>
    </tr>
    <tr>
      <th>545.0</th>
      <td>0.96097</td>
      <td>0.96097</td>
      <td>0.67896</td>
      <td>0.42821</td>
      <td>0.39055</td>
      <td>0.36129</td>
      <td>0.33766</td>
      <td>0.31800</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.23705</td>
      <td>0.19650</td>
      <td>0.17730</td>
      <td>0.17090</td>
      <td>0.15291</td>
      <td>0.13934</td>
      <td>0.13431</td>
      <td>0.12151</td>
      <td>0.11770</td>
      <td>0.11221</td>
      <td>0.10779</td>
      <td>0.10550</td>
      <td>0.10428</td>
      <td>0.10382</td>
      <td>0.10535</td>
      <td>0.10474</td>
      <td>0.11160</td>
      <td>0.11068</td>
      <td>0.11160</td>
      <td>0.11419</td>
      <td>0.11602</td>
      <td>0.11678</td>
      <td>0.11602</td>
      <td>0.11434</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 34 columns</p>
</div>




```python
ivol_array = ivol_df.to_numpy()
ivol_array
```




    array([[1.14335, 1.19435, 0.8474 , ...,     nan,     nan,     nan],
           [1.12623, 1.17682, 0.83475, ...,     nan,     nan,     nan],
           [1.10911, 1.15929, 0.8221 , ...,     nan,     nan,     nan],
           ...,
           [0.8532 , 0.8532 , 0.60259, ..., 0.11892, 0.11785, 0.11434],
           [0.90746, 0.90746, 0.64101, ..., 0.118  , 0.11693, 0.11404],
           [0.96097, 0.96097, 0.67896, ..., 0.11678, 0.11602, 0.11434]])




```python
x = np.arange(0, ivol_array.shape[1])
y = np.arange(0, ivol_array.shape[0])
#mask invalid values
array = np.ma.masked_invalid(ivol_array)
xx, yy = np.meshgrid(x, y)
#get only the valid values
x1 = xx[~array.mask]
y1 = yy[~array.mask]
newarr = array[~array.mask]

GD1 = interpolate.griddata((x1, y1), newarr.ravel(),
                          (xx, yy),
                            method='cubic')

GD1
```




    array([[1.14335, 1.19435, 0.8474 , ...,     nan,     nan,     nan],
           [1.12623, 1.17682, 0.83475, ...,     nan,     nan,     nan],
           [1.10911, 1.15929, 0.8221 , ...,     nan,     nan,     nan],
           ...,
           [0.8532 , 0.8532 , 0.60259, ..., 0.11892, 0.11785, 0.11434],
           [0.90746, 0.90746, 0.64101, ..., 0.118  , 0.11693, 0.11404],
           [0.96097, 0.96097, 0.67896, ..., 0.11678, 0.11602, 0.11434]])




```python
vol_surf = pd.DataFrame(
    ivol_array,
    index = K,
    columns = T
).copy()

vol_surf
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0.0</th>
      <th>1.0</th>
      <th>2.0</th>
      <th>5.0</th>
      <th>6.0</th>
      <th>7.0</th>
      <th>8.0</th>
      <th>9.0</th>
      <th>13.0</th>
      <th>14.0</th>
      <th>16.0</th>
      <th>23.0</th>
      <th>28.0</th>
      <th>30.0</th>
      <th>37.0</th>
      <th>44.0</th>
      <th>57.0</th>
      <th>72.0</th>
      <th>85.0</th>
      <th>107.0</th>
      <th>118.0</th>
      <th>149.0</th>
      <th>170.0</th>
      <th>177.0</th>
      <th>261.0</th>
      <th>271.0</th>
      <th>352.0</th>
      <th>363.0</th>
      <th>380.0</th>
      <th>443.0</th>
      <th>534.0</th>
      <th>716.0</th>
      <th>744.0</th>
      <th>1080.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>402.0</th>
      <td>1.14335</td>
      <td>1.19435</td>
      <td>0.84740</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.36052</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.27272</td>
      <td>NaN</td>
      <td>0.24284</td>
      <td>0.20458</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.18309</td>
      <td>NaN</td>
      <td>0.16449</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>403.0</th>
      <td>1.12623</td>
      <td>1.17682</td>
      <td>0.83475</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.35748</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.27013</td>
      <td>NaN</td>
      <td>0.24071</td>
      <td>0.20382</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.18248</td>
      <td>NaN</td>
      <td>0.16465</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>404.0</th>
      <td>1.10911</td>
      <td>1.15929</td>
      <td>0.82210</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.35199</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.26769</td>
      <td>NaN</td>
      <td>0.23858</td>
      <td>0.20306</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.18218</td>
      <td>NaN</td>
      <td>0.16465</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>405.0</th>
      <td>1.09199</td>
      <td>1.14176</td>
      <td>0.82408</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.45092</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.34864</td>
      <td>0.30702</td>
      <td>0.27150</td>
      <td>0.28568</td>
      <td>0.27044</td>
      <td>0.26510</td>
      <td>0.24803</td>
      <td>0.23720</td>
      <td>0.20138</td>
      <td>0.18782</td>
      <td>0.18599</td>
      <td>0.19300</td>
      <td>0.18675</td>
      <td>0.18157</td>
      <td>0.16968</td>
      <td>0.16449</td>
      <td>0.16068</td>
      <td>0.15398</td>
      <td>0.14681</td>
      <td>0.14742</td>
      <td>0.13462</td>
      <td>0.09727</td>
      <td>0.00293</td>
      <td>0.00202</td>
    </tr>
    <tr>
      <th>406.0</th>
      <td>1.07487</td>
      <td>1.12438</td>
      <td>0.81143</td>
      <td>0.35829</td>
      <td>0.30711</td>
      <td>0.26872</td>
      <td>0.46327</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.34315</td>
      <td>NaN</td>
      <td>0.26845</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.26251</td>
      <td>0.24605</td>
      <td>0.23538</td>
      <td>0.20047</td>
      <td>NaN</td>
      <td>0.18507</td>
      <td>0.19239</td>
      <td>NaN</td>
      <td>0.18111</td>
      <td>NaN</td>
      <td>0.16434</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>525.0</th>
      <td>0.74192</td>
      <td>0.74192</td>
      <td>0.52394</td>
      <td>0.33019</td>
      <td>0.30108</td>
      <td>0.27851</td>
      <td>0.26022</td>
      <td>0.24498</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.18218</td>
      <td>0.15077</td>
      <td>0.13584</td>
      <td>0.13080</td>
      <td>0.11678</td>
      <td>0.11846</td>
      <td>0.11084</td>
      <td>0.10611</td>
      <td>0.10367</td>
      <td>0.10199</td>
      <td>0.10199</td>
      <td>0.10504</td>
      <td>0.10581</td>
      <td>0.10535</td>
      <td>0.11023</td>
      <td>0.10977</td>
      <td>0.11770</td>
      <td>0.11663</td>
      <td>0.11648</td>
      <td>0.12029</td>
      <td>0.12166</td>
      <td>0.12105</td>
      <td>0.11968</td>
      <td>0.11419</td>
    </tr>
    <tr>
      <th>530.0</th>
      <td>0.79801</td>
      <td>0.79801</td>
      <td>0.56372</td>
      <td>0.35534</td>
      <td>0.32409</td>
      <td>0.29970</td>
      <td>0.28004</td>
      <td>0.26373</td>
      <td>NaN</td>
      <td>0.21022</td>
      <td>0.19620</td>
      <td>0.16251</td>
      <td>0.14635</td>
      <td>0.14117</td>
      <td>0.12608</td>
      <td>0.12776</td>
      <td>0.11419</td>
      <td>0.11007</td>
      <td>0.10596</td>
      <td>0.10367</td>
      <td>0.10291</td>
      <td>0.10459</td>
      <td>0.10474</td>
      <td>0.10443</td>
      <td>0.10901</td>
      <td>0.10809</td>
      <td>0.11587</td>
      <td>0.11480</td>
      <td>0.11617</td>
      <td>0.11861</td>
      <td>0.12029</td>
      <td>0.12013</td>
      <td>0.11526</td>
      <td>0.11449</td>
    </tr>
    <tr>
      <th>535.0</th>
      <td>0.85320</td>
      <td>0.85320</td>
      <td>0.60259</td>
      <td>0.38004</td>
      <td>0.34650</td>
      <td>0.32059</td>
      <td>0.29955</td>
      <td>0.28217</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.21007</td>
      <td>0.17394</td>
      <td>0.15687</td>
      <td>0.15123</td>
      <td>0.13523</td>
      <td>0.12303</td>
      <td>0.12227</td>
      <td>0.11434</td>
      <td>0.10992</td>
      <td>0.10504</td>
      <td>0.10398</td>
      <td>0.10428</td>
      <td>0.10413</td>
      <td>0.10352</td>
      <td>0.10763</td>
      <td>0.10703</td>
      <td>0.11434</td>
      <td>0.11327</td>
      <td>0.11480</td>
      <td>0.11693</td>
      <td>0.11876</td>
      <td>0.11892</td>
      <td>0.11785</td>
      <td>0.11434</td>
    </tr>
    <tr>
      <th>540.0</th>
      <td>0.90746</td>
      <td>0.90746</td>
      <td>0.64101</td>
      <td>0.40427</td>
      <td>0.36876</td>
      <td>0.34101</td>
      <td>0.31876</td>
      <td>0.30016</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.22364</td>
      <td>0.18538</td>
      <td>0.16708</td>
      <td>0.16114</td>
      <td>0.14422</td>
      <td>0.13126</td>
      <td>0.13050</td>
      <td>0.11968</td>
      <td>0.11267</td>
      <td>0.10763</td>
      <td>0.10703</td>
      <td>0.10428</td>
      <td>0.10428</td>
      <td>0.10337</td>
      <td>0.10642</td>
      <td>0.10581</td>
      <td>0.11297</td>
      <td>0.11206</td>
      <td>0.11190</td>
      <td>0.11556</td>
      <td>0.11739</td>
      <td>0.11800</td>
      <td>0.11693</td>
      <td>0.11404</td>
    </tr>
    <tr>
      <th>545.0</th>
      <td>0.96097</td>
      <td>0.96097</td>
      <td>0.67896</td>
      <td>0.42821</td>
      <td>0.39055</td>
      <td>0.36129</td>
      <td>0.33766</td>
      <td>0.31800</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.23705</td>
      <td>0.19650</td>
      <td>0.17730</td>
      <td>0.17090</td>
      <td>0.15291</td>
      <td>0.13934</td>
      <td>0.13431</td>
      <td>0.12151</td>
      <td>0.11770</td>
      <td>0.11221</td>
      <td>0.10779</td>
      <td>0.10550</td>
      <td>0.10428</td>
      <td>0.10382</td>
      <td>0.10535</td>
      <td>0.10474</td>
      <td>0.11160</td>
      <td>0.11068</td>
      <td>0.11160</td>
      <td>0.11419</td>
      <td>0.11602</td>
      <td>0.11678</td>
      <td>0.11602</td>
      <td>0.11434</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 34 columns</p>
</div>




```python
vol_surf = ivol_df.loc[:,ivol_df.columns>0].dropna(how='any', axis=1).copy()
K = vol_surf.index.tolist()
T = vol_surf.columns.tolist()
vol_surf
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1.0</th>
      <th>2.0</th>
      <th>16.0</th>
      <th>44.0</th>
      <th>72.0</th>
      <th>85.0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>402.0</th>
      <td>1.19435</td>
      <td>0.84740</td>
      <td>0.36052</td>
      <td>0.27272</td>
      <td>0.24284</td>
      <td>0.20458</td>
    </tr>
    <tr>
      <th>403.0</th>
      <td>1.17682</td>
      <td>0.83475</td>
      <td>0.35748</td>
      <td>0.27013</td>
      <td>0.24071</td>
      <td>0.20382</td>
    </tr>
    <tr>
      <th>404.0</th>
      <td>1.15929</td>
      <td>0.82210</td>
      <td>0.35199</td>
      <td>0.26769</td>
      <td>0.23858</td>
      <td>0.20306</td>
    </tr>
    <tr>
      <th>405.0</th>
      <td>1.14176</td>
      <td>0.82408</td>
      <td>0.34864</td>
      <td>0.26510</td>
      <td>0.23720</td>
      <td>0.20138</td>
    </tr>
    <tr>
      <th>406.0</th>
      <td>1.12438</td>
      <td>0.81143</td>
      <td>0.34315</td>
      <td>0.26251</td>
      <td>0.23538</td>
      <td>0.20047</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>525.0</th>
      <td>0.74192</td>
      <td>0.52394</td>
      <td>0.18218</td>
      <td>0.11846</td>
      <td>0.10611</td>
      <td>0.10367</td>
    </tr>
    <tr>
      <th>530.0</th>
      <td>0.79801</td>
      <td>0.56372</td>
      <td>0.19620</td>
      <td>0.12776</td>
      <td>0.11007</td>
      <td>0.10596</td>
    </tr>
    <tr>
      <th>535.0</th>
      <td>0.85320</td>
      <td>0.60259</td>
      <td>0.21007</td>
      <td>0.12303</td>
      <td>0.11434</td>
      <td>0.10992</td>
    </tr>
    <tr>
      <th>540.0</th>
      <td>0.90746</td>
      <td>0.64101</td>
      <td>0.22364</td>
      <td>0.13126</td>
      <td>0.11968</td>
      <td>0.11267</td>
    </tr>
    <tr>
      <th>545.0</th>
      <td>0.96097</td>
      <td>0.67896</td>
      <td>0.23705</td>
      <td>0.13934</td>
      <td>0.12151</td>
      <td>0.11770</td>
    </tr>
  </tbody>
</table>
<p>108 rows × 6 columns</p>
</div>




```python
vol_matrix = ql.Matrix(len(K),len(T),0.0)
for i,k in enumerate(K):
    for j,t in enumerate(T):
        vol_matrix[i][j] = float(vol_surf.loc[k,t])

print(vol_matrix)
```

    | 1.19435 0.8474 0.36052 0.27272 0.24284 0.20458 |
    | 1.17682 0.83475 0.35748 0.27013 0.24071 0.20382 |
    | 1.15929 0.8221 0.35199 0.26769 0.23858 0.20306 |
    | 1.14176 0.82408 0.34864 0.2651 0.2372 0.20138 |
    | 1.12438 0.81143 0.34315 0.26251 0.23538 0.20047 |
    | 1.10685 0.79862 0.33979 0.25977 0.23324 0.19955 |
    | 1.08947 0.78597 0.33446 0.25717 0.23141 0.19818 |
    | 1.07209 0.77332 0.33095 0.25443 0.22943 0.19711 |
    | 1.05472 0.76067 0.33141 0.25245 0.2276 0.1965 |
    | 1.03734 0.74817 0.32211 0.24909 0.22592 0.19498 |
    | 1.02011 0.73552 0.31861 0.24635 0.22364 0.19376 |
    | 1.00273 0.72302 0.31312 0.24406 0.22242 0.19285 |
    | 0.98551 0.71036 0.30961 0.24193 0.22028 0.19147 |
    | 0.96828 0.69786 0.30946 0.23964 0.2183 0.19056 |
    | 0.95106 0.68536 0.30047 0.23675 0.21647 0.18949 |
    | 0.93383 0.67286 0.29681 0.23446 0.21464 0.18843 |
    | 0.91661 0.66037 0.29147 0.23202 0.21312 0.18721 |
    | 0.89938 0.6593 0.28766 0.22958 0.21129 0.18629 |
    | 0.88231 0.65701 0.28705 0.22714 0.20961 0.18492 |
    | 0.86524 0.64436 0.27836 0.22455 0.20763 0.18385 |
    | 0.84801 0.63171 0.27455 0.22196 0.20596 0.18279 |
    | 0.83094 0.61906 0.26922 0.21983 0.20428 0.18157 |
    | 0.81387 0.60625 0.26525 0.21769 0.20245 0.18065 |
    | 0.79679 0.5936 0.26556 0.21495 0.20093 0.17958 |
    | 0.77972 0.58095 0.25733 0.21266 0.19894 0.17836 |
    | 0.76265 0.56829 0.25184 0.21038 0.19727 0.17715 |
    | 0.74558 0.55579 0.24772 0.20794 0.19589 0.17623 |
    | 0.7285 0.54314 0.24498 0.20596 0.19407 0.17486 |
    | 0.72652 0.53049 0.24315 0.20382 0.19224 0.1741 |
    | 0.7093 0.51784 0.23644 0.20123 0.19071 0.17288 |
    | 0.69192 0.50519 0.23217 0.19925 0.18904 0.17166 |
    | 0.66052 0.49269 0.22669 0.19696 0.18751 0.17059 |
    | 0.64344 0.48003 0.22349 0.19498 0.18568 0.16937 |
    | 0.62637 0.46738 0.22227 0.19285 0.18431 0.1683 |
    | 0.60945 0.45488 0.21556 0.19086 0.18248 0.16708 |
    | 0.60549 0.44955 0.21099 0.18736 0.18096 0.16587 |
    | 0.58826 0.43674 0.20641 0.18538 0.17943 0.1648 |
    | 0.57089 0.42409 0.20275 0.18324 0.17776 0.16388 |
    | 0.55366 0.41769 0.20169 0.18141 0.17623 0.16251 |
    | 0.53628 0.40473 0.19513 0.17928 0.17471 0.1616 |
    | 0.52958 0.39177 0.19102 0.17745 0.17303 0.16022 |
    | 0.51189 0.37882 0.18705 0.17562 0.17166 0.15931 |
    | 0.49436 0.36586 0.18431 0.17349 0.16998 0.15824 |
    | 0.47668 0.3529 0.18004 0.17166 0.16846 0.15702 |
    | 0.45885 0.35016 0.17623 0.16968 0.16693 0.1558 |
    | 0.44116 0.32684 0.17303 0.16769 0.16526 0.15474 |
    | 0.43156 0.3279 0.16968 0.16602 0.16358 0.15352 |
    | 0.41357 0.31007 0.16602 0.16404 0.16251 0.15245 |
    | 0.39543 0.32897 0.1651 0.16251 0.16114 0.15108 |
    | 0.37714 0.2744 0.15931 0.16053 0.15916 0.15016 |
    | 0.35885 0.27364 0.15718 0.1587 0.15809 0.14894 |
    | 0.34056 0.25992 0.15413 0.15702 0.15626 0.14773 |
    | 0.32836 0.24605 0.15077 0.15535 0.15458 0.14666 |
    | 0.29711 0.22867 0.14818 0.15367 0.15306 0.14544 |
    | 0.27867 0.21495 0.14498 0.15184 0.15215 0.14422 |
    | 0.26602 0.20413 0.14285 0.15016 0.15001 0.143 |
    | 0.25717 0.193 0.1401 0.14849 0.14818 0.14193 |
    | 0.23294 0.1837 0.13812 0.14681 0.14666 0.14071 |
    | 0.21754 0.17333 0.13584 0.14544 0.14529 0.13965 |
    | 0.19742 0.16602 0.1334 0.1433 0.14376 0.13843 |
    | 0.18065 0.16007 0.13096 0.14178 0.14224 0.13721 |
    | 0.16556 0.15428 0.12898 0.1401 0.14117 0.13538 |
    | 0.15352 0.1491 0.12715 0.13843 0.13904 0.13416 |
    | 0.14376 0.14544 0.12501 0.1369 0.13766 0.1337 |
    | 0.13477 0.14285 0.12288 0.13507 0.13629 0.13233 |
    | 0.13126 0.14178 0.1209 0.13355 0.13492 0.13111 |
    | 0.12715 0.13965 0.11892 0.13202 0.1334 0.13004 |
    | 0.12456 0.13843 0.11724 0.1305 0.13218 0.12882 |
    | 0.12196 0.13766 0.11556 0.12898 0.1308 0.12776 |
    | 0.1209 0.1369 0.11419 0.12745 0.12928 0.12669 |
    | 0.12212 0.13538 0.11251 0.12608 0.12806 0.12547 |
    | 0.12044 0.13568 0.11114 0.12456 0.12669 0.1244 |
    | 0.12456 0.13523 0.10992 0.12303 0.12547 0.12334 |
    | 0.1334 0.13324 0.10885 0.12181 0.12425 0.12242 |
    | 0.12349 0.13523 0.10794 0.12044 0.12318 0.1212 |
    | 0.13843 0.13355 0.10672 0.11892 0.12196 0.12029 |
    | 0.15306 0.1302 0.10611 0.1177 0.1209 0.11922 |
    | 0.16754 0.14209 0.1052 0.11648 0.11968 0.11831 |
    | 0.18187 0.14742 0.10489 0.11541 0.11861 0.11739 |
    | 0.19605 0.13782 0.10413 0.11434 0.11754 0.11632 |
    | 0.21007 0.14773 0.10367 0.11343 0.11648 0.11526 |
    | 0.22394 0.15763 0.10291 0.11236 0.11556 0.11449 |
    | 0.23766 0.16739 0.10367 0.11145 0.11449 0.11358 |
    | 0.25138 0.17699 0.10352 0.11068 0.11373 0.11282 |
    | 0.26495 0.1866 0.10382 0.10977 0.11282 0.1119 |
    | 0.27836 0.1962 0.10321 0.10901 0.1119 0.11129 |
    | 0.29162 0.20565 0.10352 0.10824 0.11114 0.11038 |
    | 0.30489 0.21495 0.10428 0.10794 0.11038 0.10977 |
    | 0.31815 0.22425 0.10428 0.10733 0.10962 0.10901 |
    | 0.33126 0.23355 0.10321 0.10657 0.10885 0.10824 |
    | 0.34421 0.24284 0.10718 0.10596 0.10824 0.10718 |
    | 0.35717 0.25199 0.10657 0.10565 0.10763 0.10657 |
    | 0.37013 0.26098 0.10459 0.1055 0.10733 0.10596 |
    | 0.38293 0.27013 0.10809 0.10504 0.10657 0.10596 |
    | 0.39559 0.27912 0.11175 0.10489 0.10611 0.10504 |
    | 0.40839 0.28812 0.11129 0.10382 0.10565 0.10459 |
    | 0.42089 0.29696 0.1148 0.10428 0.1052 0.10443 |
    | 0.43354 0.30595 0.11831 0.10382 0.10459 0.10398 |
    | 0.44604 0.31464 0.12181 0.10413 0.10443 0.10337 |
    | 0.50763 0.35824 0.12349 0.10382 0.10276 0.10169 |
    | 0.56784 0.40092 0.13858 0.10581 0.10245 0.10078 |
    | 0.62698 0.44269 0.15337 0.11007 0.10245 0.10047 |
    | 0.68491 0.48369 0.168 0.11251 0.10367 0.10138 |
    | 0.74192 0.52394 0.18218 0.11846 0.10611 0.10367 |
    | 0.79801 0.56372 0.1962 0.12776 0.11007 0.10596 |
    | 0.8532 0.60259 0.21007 0.12303 0.11434 0.10992 |
    | 0.90746 0.64101 0.22364 0.13126 0.11968 0.11267 |
    | 0.96097 0.67896 0.23705 0.13934 0.12151 0.1177 |
    
    


```python
bicubic_vol = ql.BicubicSpline(T,K,vol_matrix)
```


```python
K = np.linspace(
    min(K),
    max(K),
    50
)
T = np.linspace(
    1,
    10,
    50
)

KK,TT = np.meshgrid(K,T)

V = np.array(
    [[bicubic_vol(float(t),float(k),False) for k in K] for t in T]
    )

plt.rcParams['figure.figsize']=(7,5)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.view_init(elev=20, azim=120)  
surf = ax.plot_surface(KK,TT,V, rstride=1, cstride=1, cmap=cm.coolwarm,
                linewidth=0.1)
fig.colorbar(surf, shrink=0.3, aspect=5)

ax.set_xlabel("Strike", size=9)
ax.set_ylabel("Maturity", size=9)
ax.set_zlabel("Volatility", size=9)

plt.tight_layout()
plt.show()
plt.cla()
plt.clf()
```


    
![png](output_13_0.png)
    



    <Figure size 700x500 with 0 Axes>



```python

```
