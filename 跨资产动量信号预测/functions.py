# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 13:19:59 2026

@author: zhangziyu
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pandas import Series, DataFrame

#定义标准化函数：把所有数值按照最大值和最小值投影到[-1,1]区间
def normalize(series):
    x_min = series.min()
    x_max = series.max()
    
    if x_max == x_min:
        print(f'警告：所有值都相同，将返回全0序列')
        normalized = pd.Series([0]*len(series), index=series.index)

    normalized = 2 * (series - x_min) / (x_max - x_min) - 1
    
    return normalized

#通用回测函数
#其中r、r_on、r_id 分别为投资标的指数的日收益率、隔夜收益率、日内收益率
def backtest(w, r, r_on, r_id):
    r_timing = w.shift(2)*r + w.diff().shift()*r_id
    r_timing.cumsum().plot()
    plt.show()
    s_evaluation = Series({
        '年化收益': r_timing.mean() * 240, 
        '年化波动': r_timing.std() * 240**0.5, 
        '夏普比率': r_timing.mean() / r_timing.std() * 240**0.5, 
        })
    print(s_evaluation)
    return r_timing


#计算投资标的的三种收益率r、r_on、r_id： 分别为投资标的指数的日收益率、隔夜收益率、日内收益率
#注意：导入的dataframe必须包含开盘价和收盘价
def calculate_three_ratios(df: pd.DataFrame):
    
    #计算标的的三种收益率
    df['daily_return'] = df['close'].pct_change()
    df['overnight_return'] = df['open'] / df['close'].shift(1) - 1
    df['intraday_return'] = df['close'] / df['open'] - 1
    
    #命名为series变量
    r = df.set_index('trade_date')['daily_return']
    r_on = df.set_index('trade_date')['overnight_return']
    r_id = df.set_index('trade_date')['intraday_return']
    
    return r, r_on, r_id

    


