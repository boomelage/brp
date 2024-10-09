import os
import sys
import time
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)
from mlop import mlop
sys.path.append(os.path.join(current_dir,'train_data'))
sys.path.append(os.path.join(
    current_dir,
    'historical_data',
    'historical_generation'))
pd.set_option("display.max_columns",None)
train_start = time.time()
train_start_datetime = datetime.fromtimestamp(train_start)
train_start_tag = train_start_datetime.strftime('%c')

print("\n"+"#"*18+"\n# training start #\n"+
      "#"*18+"\n"+f"\n{train_start_tag}\n")

"""
# =============================================================================
                                importing data
"""

from HDF_collection import contracts

dataset = contracts.copy()
mlop = mlop(user_dataset = dataset)

