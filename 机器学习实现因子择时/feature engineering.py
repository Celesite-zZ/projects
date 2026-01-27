# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 15:13:28 2026

@author: zhangziyu
"""


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pandas import Series, DataFrame

import functions

from functions import factors_list,r_list





#%%导入自变量数据

x_data = pd.read_csv('x_data.csv', index_col=0, parse_dates=['trade_date'])


#%%时间特征

x_data_1 = functions.add_time_features_from_datetime_index(x_data)


#%%动量特征

x_data_2 = functions.add_momentum_features(x_data_1)


#%%储存数据



x_data_2.to_csv('X.csv', index=True)
print("done")

















