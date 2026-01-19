# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 10:47:20 2026

@author: zhangziyu
"""


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pandas import Series, DataFrame

from functions import backtest
from functions import calculate_three_ratios


#%%读取和处理数据

df = pd.read_excel('指数日线_932000_CSI_20120101_None.xlsx', parse_dates=['trade_date'])


#%% 因子计算和信号获取

#一
#5日乖离率（动量类因子）：
#（收盘价-收盘价的N日简单平均）/ 收盘价的N日简单平均*100，在此n取5

close = df.set_index('trade_date')['close']

signal_raw1 = (close - close.rolling(5).mean()) / close.rolling(5).mean() * 100

w1 = np.sign(signal_raw1)


#二
#6日成交金额的移动平均值（情绪类因子）：
#成交金额增加 => 市场情绪积极 => 做多

amount = df.set_index('trade_date')['amount']

signal_raw2 = amount.rolling(6).mean().pct_change()

w2 = np.sign(signal_raw2)


#三
#12日变动率ROC（动量类因子）：
#A = 今天的收盘价—12天前的收盘价
#B = 12天前的收盘价
#ROC = A / B * 100

close_lag = close.shift(12)
ROC = (close - close_lag) / close_lag * 100

w3 = np.sign(ROC)


#四
#12日平均成交量的移动平均值（情绪类因子）：
#成交量上涨 => 市场情绪积极 => 做多

vol = df.set_index('trade_date')['vol']

signal_raw4 = vol.rolling(12).mean().pct_change()

w4 = np.sign(signal_raw4)

#五
#6日均幅指标（情绪类因子）：
#真实振幅的6日移动平均
#振幅扩大 => 市场震荡剧烈 => 做空

change = df.set_index('trade_date')['change']

signal_raw5 = change.rolling(6).mean().pct_change()

w5 = -np.sign(signal_raw5)




#%% 回测

r, r_on, r_id = calculate_three_ratios(df)

print('5日乖离率回测情况：')
backtest(w1, r, r_on, r_id)

print('6日成交金额移动平均回测情况：')
backtest(w2, r, r_on, r_id)

print('ROC回测情况：')
backtest(w3, r, r_on, r_id)


print('6日成交量移动平均回测情况：')
backtest(w4, r, r_on, r_id)

print('6日均幅指标情况：')
backtest(w5, r, r_on, r_id)
























