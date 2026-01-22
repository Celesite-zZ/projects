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
def backtest_fundamental(w, r, r_on, r_id):
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

def backtest_multi(w, r, r_on, r_id):
    r_timing = w.shift(2).mul(r, 0) + w.shift().diff().mul(r_id, 0)
    r_timing.cumsum().plot()
    plt.show()
    df_evaluation = DataFrame({
        '年化收益': r_timing.mean() * 240, 
        '年化波动': r_timing.std() * 240**0.5, 
        '夏普比率': r_timing.mean() / r_timing.std() * 240**0.5, 
        })
    
    #items方法返回index和value的元组
    '''
    for idx, value in s_evaluation.items():
        print(idx)
        print(value)
        '''
      
    print(df_evaluation)    
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


#计算两个df的收益率之差，即多空对冲信号 
#注意产生NaN值后相减会吞掉数据  
def calculate_diff_r(df1: pd.DataFrame, df2: pd.DataFrame):
    r1, r_on_1, r_id_1 = calculate_three_ratios(df1)
    r2, r_on_2, r_id_2 = calculate_three_ratios(df2)
    
    return r1-r2, r_on_1-r_on_2, r_id_1-r_id_2

#输出收益率的不同时间滚动周期的移动平均线，即时间序列动量
def signal_multi_time_mom(r):
    w = np.sign(DataFrame({
    'n=5': r.rolling(5).mean(), 
    'n=20': r.rolling(20).mean(), 
    'n=60': r.rolling(60).mean(), 
    'n=240': r.rolling(240).mean(), 
    }))
    
    return w

