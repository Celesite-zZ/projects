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

#定义标准化函数
def normalize(series):
    x_min = series.min()
    x_max = series.max()
    
    if x_max == x_min:
        print(f'警告：所有值都相同，将返回全0序列')
        normalized = pd.Series([0]*len(series), index=series.index)

    normalized = 2 * (series - x_min) / (x_max - x_min) - 1
    
    return normalized

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







