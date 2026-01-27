# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 16:05:43 2026

@author: zhangziyu
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pandas import Series, DataFrame


data = pd.read_excel('y.xlsx')

#日期数据填充
data['trade_date'] = data['trade_date'].ffill()


# 提取MKT和SMB的数据作为因变量
mkt_data = data[data['fut_code'] == 'MKT'][['trade_date', 'r']].copy()
mkt_data.rename(columns={'r': 'MKT_r'}, inplace=True)
smb_data = data[data['fut_code'] == 'SMB'][['trade_date', 'r']].copy()
smb_data.rename(columns={'r': 'SMB_r'}, inplace=True)

# 设置日期为索引
mkt_data.set_index('trade_date', inplace=True)
smb_data.set_index('trade_date', inplace=True)


mkt_data.to_csv('y_MKT.csv', index=True)
smb_data.to_csv('y_SMB.csv', index=True)





