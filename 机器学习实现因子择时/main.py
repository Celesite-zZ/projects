# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 11:01:37 2026

@author: zhangziyu
"""




import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pandas import Series, DataFrame

import functions

from functions import factors_list,r_list

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


import seaborn as sns


#%%数据导入和整理

X = pd.read_csv('X.csv')

y_MKT = pd.read_csv('y_MKT.csv')
y_SMB = pd.read_csv('y_SMB.csv')


X = X.set_index('trade_date')


#去掉特征中的MKT相关列
MKT_features = mkt_features = [col for col in X.columns if 'MKT' in col]
X_data = X.drop(columns=MKT_features)


y_MKT = Series(y_MKT.set_index('trade_date')['MKT_r'])




#%%模型训练

train_window = 500 #滚动时间训练窗口
test_window = 15 #预测窗口
step_size = 15 #步长

predictions, actuals, insample_r2, outsample_r2 = functions.rolling_window_train_predict(
    X_data, y_MKT, 
    Series(X.index), 
    train_size=train_window, 
    test_size=test_window, 
    step=step_size
)




#%%模型评估和可视化

#m模型评估
functions.evaluate_regression_model(predictions, actuals)


#可视化
functions.plot_regression_results(predictions, actuals)

#样本内外R方
print(f'样本内平均R方：')
print(f'{np.mean(insample_r2)}')
print(f'样本外平均R方：')
print(f'{np.mean(outsample_r2)}')


#%%累计收益曲线


#数据对齐
aligned_data = functions.align_predictions_with_data(predictions, X['MKT_r'], X['MKT_r_on'], X['MKT_r_id'])

w = np.sign(aligned_data['predictions'])


functions.backtest_fundamental(
        w = w,
        r = aligned_data['r'],
        r_on = aligned_data['r_on'],
        r_id = aligned_data['r_id']
    )









