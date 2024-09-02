import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
import numpy as np
from settings import random_state
np.random.seed(random_state)

# =============================================================================
                                                            # Generate features
# =============================================================================
# from AdataGeneration import generate_features, generate_qldates_grid
# features, feature_names = generate_features()
# features.to_csv(r'features.csv')
# features = generate_qldates_grid(features)
# print(r"Features generated!")
# =============================================================================
                                                       # Generate test features
# =============================================================================
# from AdataGeneration import generate_small_features, generate_qldates_grid
# features, feature_names = generate_small_features()
# features.to_csv(r'features.csv')
# features = generate_qldates_grid(features)
# print(r"Features generated!")
# =============================================================================

# =============================================================================
                                                              # Import features
# =============================================================================
# import pandas as pd
# from AdataGeneration import generate_qldates_grid
# features = pd.read_csv(r'features.csv')
# features = features.iloc[:,1:]
# features = generate_qldates_grid(features)
# =============================================================================

# =============================================================================
                                                             # Price securities
# =============================================================================
# from Bpricing import BS_price_vanillas
# from Bpricing import heston_price_vanillas
# import concurrent.futures
# def run_in_parallel(features):
#     with concurrent.futures.ThreadPoolExecutor() as executor:
#         future_bs = executor.submit (BS_price_vanillas, 
#                                     features)
#         future_heston = executor.submit(heston_price_vanillas, 
#                                         features)
#         
#         BS_vanillas = future_bs.result()
#         heston_vanillas = future_heston.result()
#         
#     return BS_vanillas, heston_vanillas
# 
# BS_vanillas, heston_vanillas = run_in_parallel(features)
# 
# print(r'Securities priced!')
# =============================================================================

# =============================================================================
                                                                # Import prices
import pandas as pd
heston_vanillas = pd.read_csv(r'heston_vanillas.csv')
heston_vanillas = heston_vanillas.iloc[:,1:]

BS_vanillas = pd.read_csv(r'BS_vanillas.csv')
BS_vanillas = BS_vanillas.iloc[:,1:]

option_prices = heston_vanillas.copy()

print(r'Data imported!')