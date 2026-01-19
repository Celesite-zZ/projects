# -*- coding: utf-8 -*-
"""
Created on Mon Jan  5 10:46:26 2026

@author: zhangziyu
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from functions import normalize
from functions import backtest


df = pd.read_excel('指数日线_000985_CSI_20120101_None_with_returns.xlsx')
df['trade_date'] = pd.to_datetime(df['trade_date'].astype(str))

# 计算所需收益率
df['daily_return'] = df['close'].pct_change()
df['overnight_return'] = df['open'] / df['close'].shift(1) - 1
df['intraday_return'] = df['close'] / df['open'] - 1


#%%交易标的的3个收益率，回测用

r = df.set_index('trade_date')['daily_return']
r_on = df.set_index('trade_date')['overnight_return']
r_id = df.set_index('trade_date')['intraday_return']



#信号生成：
#三个因子的简明代理：
#市场因子r1（中证全指） —— 000985.SCI —— 类似于动量趋势信号
#价值因子r2（现在/未来：中证红利/科创50） —— 000922.SCI/000688.SH ——类似于市场情绪信号
#规模因子r3（小盘/大盘：中证2000/沪深300） —— 932000.SCI/399300.SZ —— 资金在小盘和大盘的流动趋势

#总体结构：把三个分数标准化r123_std，然后计算总体得分，通过总体得分去生成信号

#以科创200为例，它和小盘、科技股关系比较大，所以市场因子和规模因子对它的上涨有正向作用
#而价值因子对它有反向作用

#%%第一部分：表格文件导入


# 列出当前目录下所有的Excel文件
excel_files = [f for f in os.listdir('.') if f.endswith('.xlsx')]

# 创建一个字典来存储所有表格,后续只需要使用这个字典即可
data_dict = {}

for file in excel_files:
    try:
        # 去掉扩展名作为字典的键
        name = os.path.splitext(file)[0]
        # 读取Excel文件
        data_dict[name] = pd.read_excel(file, parse_dates=['trade_date'])
        print(f"成功导入: {file}")
    except Exception as e:
        print(f"导入失败 {file}: {e}")

print(f"\n总共导入了 {len(data_dict)} 个Excel文件")

data_dict[name]


#%%第二部分：信号准备

###r1
r1_raw = data_dict['指数日线_000985_CSI_20120101_None_with_returns'].set_index('trade_date')['close']

r1 = r1_raw.pct_change() #表示大盘的短期动量趋势

#r1.hist()
#plt.show()

###r2
r2_value = data_dict['指数日线_000922_CSI_20120101_None_with_returns'].set_index('trade_date')['close'] 
r2_growth = data_dict['指数日线_000688_SH_20120101_None_with_returns'].set_index('trade_date')['close']

r2_value = r2_value.pct_change()
r2_growth = r2_growth.pct_change()

hml = r2_value - r2_growth
 
#r2_raw = r2_now / r2_future

#r2 = (r2_raw - r2_raw.shift()) / r2_raw.shift()


###r3

r3_small = data_dict['指数日线_932000_CSI_20120101_None_with_returns'].set_index('trade_date')['open']   
r3_large =  data_dict['指数日线_399300_SZ_20120101_None_with_returns'].set_index('trade_date')['open'] 

r3_small = r3_small.pct_change()
r3_large = r3_large.pct_change()

smb = r3_small - r3_large


#r3_raw = r3_small / r3_large

#r3 = (r3_raw - r3_raw.shift()) / r3_raw.shift()





#%%第三部分：信号生成


#信号生成：把三个比率线性相加

'''
alpha1,alpha2,alpha3 = 0.4,0.3,0.3

signal = alpha1*r1_common + alpha2*r2_common - alpha3*r3_common
'''


w = np.sign(hml.rolling(20).mean())

backtest( w, r, r_on, r_id)





