# -*- coding: utf-8 -*-
"""
Created on Mon Dec 29 15:38:04 2025

@author: zhangziyu
"""

#%%
import pandas as pd
import numpy as np

import Tushare_data_loader
import data_processing

#获取原始指数数据
Tushare_data_loader.fetch_and_save_index_daily(
        ts_code="399300.SZ",
        start_date="20120101",
        end_date= None,
        excel_name= None
    )


#%%
#计算三种收益率（可选）

file_name = Tushare_data_loader.file_name
data_processing.calculate_returns(file_name)


#%%

#获取期货日线数据

Tushare_data_loader.fetch_and_save_fut_daily(
        ts_code="IF.CFX",
        start_date="20120101",
        end_date= None,
        excel_name= None    
    
    )



