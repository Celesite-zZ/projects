# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 19:22:48 2026

@author: zhangziyu
"""


import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from pandas import Series, DataFrame

import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

import seaborn as sns

#创建因子列表，方便后期调整

factors_list = [
        'BAB', 'CMA', 'HML','MKT', 'MOM', 'RMW', 'SMB', 'STR',
        'CRUDE', 'GOLD', 'METAL', 'SLOPE', 'STEEL', 'TERM',
        '周期', '工业', '消费', '科技', '金融'
    ]

r_list = ['r', 'r_on', 'r_id']


#添加时间相关特征
def add_time_features(df: DataFrame):
    
    # 基本时间特征
    df1['day_of_week'] = df1['trade_date'].dt.dayofweek  # 周一=0
    df1['month'] = df1['trade_date'].dt.month
    df1['quarter'] = df1['trade_date'].dt.quarter
    df1['day_of_month'] = df1['trade_date'].dt.day
    
 
    # 月初效应（前5个交易日）
    df1['is_first_week'] = (df1['day_of_month'] <= 5).astype(int)
    
    # 月末效应（最后5个交易日）
    days_in_month = df1['trade_date'].dt.days_in_month
    df1['is_last_week'] = (df1['day_of_month'] > days_in_month - 5).astype(int)

        
    return df


#从时间索引（DatetimeIndex）中添加时间特征
def add_time_features_from_datetime_index(df: DataFrame):
    
    # 创建新的DataFrame副本
    df_new = df.copy()
    
    # 1. 基本时间特征
    df_new['day_of_week'] = df.index.dayofweek  
    df_new['month'] = df.index.month
    df_new['quarter'] = df.index.quarter
    df_new['year'] = df.index.year
    
    # 2. 周数（使用isocalendar()方法）
    # isocalendar()返回一个DataFrame，所以需要单独处理
    iso_calendar = df.index.isocalendar()
    df_new['week_of_year'] = iso_calendar.week
    df_new['week_year'] = iso_calendar.year  # ISO年份
    
    # 3. 特殊日期标记:月份、季度、年的开始和末尾
    df_new['is_month_start'] = df.index.is_month_start.astype(int)
    df_new['is_month_end'] = df.index.is_month_end.astype(int)
    df_new['is_quarter_start'] = df.index.is_quarter_start.astype(int)
    df_new['is_quarter_end'] = df.index.is_quarter_end.astype(int)
    df_new['is_year_start'] = df.index.is_year_start.astype(int)
    df_new['is_year_end'] = df.index.is_year_end.astype(int)
    
    # 4. 月初/月末效应
    days_in_month = df.index.days_in_month  # 注意：是属性，不是方法
    df_new['is_first_week'] = (df.index.day <= 5).astype(int)
    df_new['is_last_week'] = (df.index.day > days_in_month - 5).astype(int)
    
    # 5. 周内效应
    df_new['is_monday'] = (df.index.dayofweek == 0).astype(int)
    df_new['is_tuesday'] = (df.index.dayofweek == 1).astype(int)
    df_new['is_wednesday'] = (df.index.dayofweek == 2).astype(int)
    df_new['is_thursday'] = (df.index.dayofweek == 3).astype(int)
    df_new['is_friday'] = (df.index.dayofweek == 4).astype(int)
       
    return df_new


#添加动量特征
def add_momentum_features(df: DataFrame):

    new_df = df.copy()
    
    #这里默认了命名形式为例如：SMB_r
    factor_cols = [f"{factor}_r" for factor in factors_list]
    
    for col in factor_cols:
        if '_r' in col:  # 只对收益率序列计算动量和反转
            # 短期动量（过去1-5天）
            new_df[f'{col}_momentum_5'] = (df[col].shift() + df[col].shift(2) + 
                                          df[col].shift(3) + df[col].shift(4) + 
                                          df[col].shift(5)).mean()
            
            # 中期动量（过去5-20天）
            new_df[f'{col}_momentum_20'] = df[col].rolling(20).mean().shift() - new_df[f'{col}_momentum_5']
            
            # 反转特征（过去涨跌幅）
            new_df[f'{col}_reversal_1'] = -df[col].shift()  # 简单反转
            
            # 动量加速度（动量的变化）
            mom_5 = df[col].rolling(5).mean()
            mom_10 = df[col].rolling(10).mean()
            new_df[f'{col}_momentum_acc'] = (mom_5.shift() - mom_10.shift())
    
    return new_df


#时间序列数据的顺序划分
def split_time_series(X, y, train_ratio=0.7, val_ratio=0.15):

    n_samples = len(X)
    
    train_end = int(n_samples * train_ratio)
    val_end = train_end + int(n_samples * val_ratio)
    
    # 训练集
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    
    # 验证集
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]
    
    # 测试集
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]
    

    
    return X_train, X_val, X_test, y_train, y_val, y_test


#XGBoost回归模型
def train_xgboost_regressor(X_train, y_train, X_test, y_test):


    
    # 定义XGBoost回归参数
    params = {
        'objective': 'reg:squarederror',  # 回归问题，使用平方误差
        'eval_metric': 'rmse',            # 评估指标：均方根误差
        'max_depth': 4,                   # 树的最大深度
        'learning_rate': 0.1,             # 学习率
        'n_estimators': 100,              # 树的数量
        'subsample': 0.8,                 # 每棵树使用的样本比例
        'colsample_bytree': 0.8,          # 每棵树使用的特征比例
        'random_state': 42,               # 随机种子
        'verbosity': 0,                   # 不输出训练过程
    }
    
    
    # 创建模型
    model = xgb.XGBRegressor(**params)
    
    # 训练模型
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_test, y_test)],
        verbose=False  # 不显示训练过程
    )
    
    print("模型训练完成")
    
    return model



#滚动时间窗口训练和预测
def rolling_window_train_predict(X, y, dates, train_size=500, test_size=20, step=20):
    """
    train_size: 训练窗口
    test_size: 预测窗口
    step: 步长
    """
    
    n_samples = len(X)
    all_predictions = []
    all_actuals = []
    all_dates = []
    
    insample_r2_list = []
    outsample_r2_list = []
    
    count = 0
    
    print(f"\n开始滚动窗口训练")
    print(f"训练窗口: {train_size}天, 预测窗口: {test_size}天, 步长: {step}天")
    
    # 循环每个窗口
    for i in range(train_size, n_samples - test_size, step):
        
        count += 1
        
        # 划分训练集和测试集（按时间顺序）
        X_train = X.iloc[i-train_size:i]
        y_train = y.iloc[i-train_size:i]
        
        X_test = X.iloc[i:i+test_size]
        y_test = y.iloc[i:i+test_size]
        
        test_dates = dates[i:i+test_size]
        
        
        # 训练模型
        model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            reg_alpha = 0.5,
            reg_lambda = 1.0,
            random_state=42,
            verbosity=0
        )
        
        model.fit(X_train, y_train)
        
        #计算样本内R方
        y_train_pred = model.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)
        insample_r2_list.append(train_r2)
        
        
        # 预测和计算样本外R方
        y_pred = model.predict(X_test)
        test_r2 = r2_score(y_test, y_pred)
        outsample_r2_list.append(test_r2)
        
        # 保存结果
        all_predictions.extend(y_pred)
        all_actuals.extend(y_test.values)
        all_dates.extend(test_dates)
        
        # 显示进度
        if i % (step * 10) == 0:
            print(f"进度: 已预测到第 {i+test_size}/{n_samples} 天")
    
    #转换为Series对象，保留日期索引
    predictions_series = pd.Series(all_predictions, index=all_dates, name='predictions')
    actuals_series = pd.Series(all_actuals, index=all_dates, name='actuals')
    
    #转换为numpy数组
    insample_r2_array = np.array(insample_r2_list)
    outsample_r2_array = np.array(outsample_r2_list)
    
    print(f'模型训练和预测全部完成')
    
    return predictions_series, actuals_series, insample_r2_array, outsample_r2_array



#模型性能评估
def evaluate_regression_model(predictions, actuals):

    print("回归模型评估结果：")
    
    
    if len(predictions) > 0:
        # 计算指标
        mse = mean_squared_error(actuals, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actuals, predictions)
        direction_acc = np.mean(np.sign(actuals) == np.sign(predictions))
        
        print(f"均方误差(MSE): {mse:.6f}")
        print(f"均方根误差(RMSE): {rmse:.6f}")
        print(f"R²分数: {r2:.4f}")
        print(f"方向准确率: {direction_acc:.4f}")
    
    return None
    


#可视化结果
def plot_regression_results(predictions, actuals):
    


    if len(predictions) > 100:
    # 只显示最后100个预测点，避免图表太拥挤
    
        show_num = 100
        plot_predictions = predictions[-show_num:]
        plot_actuals = actuals[-show_num:]
    
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
        # 图1：预测 vs 实际
        ax1.plot(plot_actuals, 'b-', label='实际值', alpha=0.7)
        ax1.plot(plot_predictions, 'r--', label='预测值', alpha=0.7)
        ax1.set_xlabel('时间（天）')
        ax1.set_ylabel('收益率')
        ax1.set_title(f'滚动窗口预测结果 (最后{show_num}天)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2：累积收益率对比
        cum_actual = np.cumprod(1 + plot_actuals) - 1
        cum_pred = np.cumprod(1 + plot_predictions) - 1
        
        ax2.plot(cum_actual, 'b-', label='实际累积收益', linewidth=2)
        ax2.plot(cum_pred, 'r--', label='预测累积收益', linewidth=2)
        ax2.set_xlabel('时间（天）')
        ax2.set_ylabel('累积收益率')
        ax2.set_title('累积收益率对比')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'MKT因子收益率 - 滚动窗口预测', fontsize=14, y=1.02)
        plt.tight_layout()
        plt.show()
        
        # 预测误差分析
        errors = plot_predictions - plot_actuals
        print(f"\n预测误差统计:")
        print(f"平均误差: {np.mean(errors):.6f}")
        print(f"误差标准差: {np.std(errors):.6f}")
        print(f"最大正误差: {np.max(errors):.6f}")
        print(f"最大负误差: {np.min(errors):.6f}")
        
        # 绘制误差分布
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
        ax.axvline(x=0, color='r', linestyle='--', linewidth=2)
        ax.set_xlabel('预测误差')
        ax.set_ylabel('频数')
        ax.set_title('预测误差分布')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
    return None



#特征重要性图
def plot_feature_importance(model, feature_names, top_n=20):

    print(f"\n分析特征重要性 (Top {top_n})...")
    
    # 获取特征重要性
    importance = model.feature_importances_
    
    # 创建特征重要性DataFrame
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # 打印Top N特征
    print(f"\nTop {top_n} 重要特征:")
    for i, row in feature_importance_df.head(top_n).iterrows():
        print(f"{row['feature']}: {row['importance']:.4f}")
    
    # 绘制特征重要性图
    plt.figure(figsize=(10, 8))
    top_features = feature_importance_df.head(top_n)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('特征重要性')
    plt.title(f'Top {top_n} 特征重要性')
    plt.gca().invert_yaxis()  # 最重要的特征在顶部
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()
    
    return feature_importance_df


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


#对齐预测的和资产标的的Series（inner join）
#注意最后返回的是一个DataFrame
def align_predictions_with_data(predictions_series, r, r_on, r_id):

    # 创建数据对齐的DataFrame
    aligned_data = pd.DataFrame({
        'predictions': predictions_series,
        'r': r,
        'r_on': r_on,
        'r_id': r_id
    })
    
    # 删除包含NaN的行（预测未覆盖的日期）
    aligned_data = aligned_data.dropna(subset=['predictions'])
    
    return aligned_data






