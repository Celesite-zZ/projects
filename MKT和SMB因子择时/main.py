# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 10:28:20 2026

@author: zhangziyu
"""

import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

import functions


#%%读取数据

#资产标的：MKT和SMB

MKT_raw = pd.read_excel('指数日线_000985_CSI_20120101_None.xlsx', parse_dates=['trade_date'])
SMB_raw_1 = pd.read_excel('指数日线_932000_CSI_20120101_None.xlsx', parse_dates=['trade_date'])
SMB_raw_2 = pd.read_excel('指数日线_399300_SZ_20120101_None.xlsx', parse_dates=['trade_date']) 

'''
df_000985 = pd.read_excel('指数日线_000985_CSI_20120101_None.xlsx', parse_dates=['trade_date'])
r_000985 = df_000985.set_index('trade_date')[[r, r_on, r_id]]
'''

df1 = pd.read_excel('指数日线_000919_SH_20120101_None.xlsx', parse_dates=['trade_date'])
df2 = pd.read_excel('指数日线_930782_CSI_20120101_None.xlsx', parse_dates=['trade_date'])
df3 = pd.read_excel('指数日线_399006_SZ_20120101_None.xlsx', parse_dates=['trade_date'])
df4 = pd.read_excel('指数日线_931375_CSI_20120101_None.xlsx', parse_dates=['trade_date'])

df_300_low_beta = pd.read_excel('指数日线_000829_CSI_20120101_None.xlsx', parse_dates=['trade_date'])
df_300_growth = pd.read_excel('指数日线_000918_CSI_20120101_None.xlsx', parse_dates=['trade_date'])


#%%数据准备

#MKT的三种收益率计算
r1, r_on_1, r_id_1 = functions.calculate_three_ratios(MKT_raw)

#用中证2000、沪深300收益率之差作为SMB的简明代理
r2, r_on_2, r_id_2 = functions.calculate_diff_r(SMB_raw_1, SMB_raw_2)

#%%信号因子准备


#HML:沪深300价值 (000919.SH) - 沪深300
#注意：HML数据只到2014年
HML,temp1,temp2 = functions.calculate_diff_r(df1, SMB_raw_2)


#BAB:300低贝 - 沪深300 
BAB,temp1,temp2 = functions.calculate_diff_r(df_300_low_beta, SMB_raw_2)

#成长因子：沪深300 - 沪深300成长 （CMA）
Growth,temp1,temp2 = functions.calculate_diff_r(SMB_raw_2, df_300_growth) 

#质量因子：300质量成长低波 (931375.CSI) - 沪深300  RMW
#注意：数据只从2020年开始
Quality,temp1,temp2 = functions.calculate_diff_r(df4, SMB_raw_2)


#利用不同的时间滚动周期捕捉长短期趋势
w1 = functions.signal_multi_time_mom(HML)
w2 = functions.signal_multi_time_mom(BAB)
w3 = functions.signal_multi_time_mom(Growth)
w4 = functions.signal_multi_time_mom(Quality)


#%%回测

print('MKT择时回测情况：')
functions.backtest_multi(w4, r1, r_on_1, r_id_1)


print('SMB择时回测情况：')
functions.backtest_multi(w4, r2, r_on_2, r_id_2)







