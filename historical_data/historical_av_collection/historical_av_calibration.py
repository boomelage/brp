from pathlib import Path
from model_settings import ms
import pandas as pd
import sys
import os
import numpy as np
from model_settings import ms
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from datetime import datetime
from datetime import timedelta
import QuantLib as ql
from itertools import product

current_dir = os.path.abspath(str(Path()))
print(current_dir)
store = pd.HDFStore(r'alphaVantage Vanillas.h5')
keys = store.keys()
len(keys)