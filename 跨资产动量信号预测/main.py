# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 10:27:48 2026

@author: zhangziyu
"""


import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt
import functions



#%%读取数据

df1 = pd.read_excel('指数日线_399300_SZ_20120101_None.xlsx', parse_dates=['trade_date'])
df2 = pd.read_excel('期货日线_IF_CFX_20120101_None.xlsx', parse_dates=['trade_date'])


#计算三种收益率
r, r_on, r_id = functions.calculate_three_ratios(df1)


#%%信号准备

#自身动量因子
#选取5日乖离率：（收盘价-收盘价的N日简单平均）/ 收盘价的N日简单平均*100，在此n取5

close = df1.set_index('trade_date')['close']

signal_raw1 = (close - close.rolling(5).mean()) / close.rolling(5).mean() * 100 

#股指期货包含了对于现货指数未来的综合预期
#期货投资者通常被认为信息更加灵敏、投资更加专业 
#=> 贴转升/升水扩大可以被视为积极的短期动量信号

close_futures = df2.set_index('trade_date')['close']

#计算基差作为升贴水程度的指标
delta = close_futures - close

signal_adjusted = delta.pct_change(periods=5)



#%%信号生成

w1 = np.sign(signal_raw1)

w2 = np.sign(signal_raw1 + signal_adjusted)


#%%回测

print('5日乖离率回测情况：')
functions.backtest(w1, r, r_on, r_id)

print('跨资产双动量因子回测情况：')
functions.backtest(w2, r, r_on, r_id)















