# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 15:13:31 2026

@author: zhangziyu
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pandas import Series, DataFrame



#%%数据导入


data = pd.read_excel('x.xlsx')
data.isnull().sum() #没有缺失值


#%%数据透视

#日期值填充
data['trade_date'] = data['trade_date'].ffill()


#提取出每个因子的三种收益率
r_data = data.pivot_table(index='trade_date', columns='factor', values='r')
r_on_data = data.pivot_table(index='trade_date', columns='factor', values='r_on')
r_id_data = data.pivot_table(index='trade_date', columns='factor', values='r_id')

#重命名列
r_data.columns = [f'{col}_r' for col in r_data.columns]
r_on_data.columns = [f'{col}_r_on' for col in r_on_data.columns]
r_id_data.columns = [f'{col}_r_id' for col in r_id_data.columns]


#合并三种收益率数据
wide_data = pd.concat([r_data, r_on_data, r_id_data], axis=1)


#%%数据处理

#应不应该填充缺失值？
#暂时先考虑不填充


#%%储存数据

#把处理好的数据储存在一个csv的文件里
wide_data.to_csv('x_data.csv', index=True)









