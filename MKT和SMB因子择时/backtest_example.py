import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import matplotlib.pyplot as plt

#%%读取数据

df = pd.read_excel('指数日线_000985_CSI_20250101_None_with_returns.xlsx', parse_dates=['trade_date'])

#%%交易标的的3个收益率，回测用

r = df.set_index('trade_date')['daily_return']
r_on = df.set_index('trade_date')['overnight_return']
r_id = df.set_index('trade_date')['intraday_return']


#%%仓位，也叫权重，信号，目标仓位。
#这里你的策略是，当天隔夜收益方向作为信号（隐含了动量效应的假设，当天隔夜收益对未来收益有正向预测作用）

w = np.sign(r_on)


#%%回测，写成函数，以后构造任意的w都能传进去回测

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

r_timing = backtest(w, r, r_on, r_id)
