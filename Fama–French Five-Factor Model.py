# -*- coding: utf-8 -*-
"""
五因子模型工程落地实现（Fama-French Five-Factor Model）
数据来源：Ken French公开数据 + Yahoo Finance (SPY股票)
步骤：1.加载因子数据 2.获取股票收益率 3.合并数据 4.线性回归拟合 5.结果分析
"""

import os
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from datetime import datetime

# === 1. 预处理：确保数据目录存在 ===
os.makedirs('data', exist_ok=True)
print("✅ 确认数据目录 'data/' 已创建")

# === 2. 下载Fama-French五因子数据（手动下载后放入data/） ===
# 请先从链接下载 F-F_5_Research_Data_Factors.csv 到 data/ 文件夹
# 如果没下载，会报错，按提示操作即可
ff_data_path = 'data/F-F_5_Research_Data_Factors.csv'
if not os.path.exists(ff_data_path):
    raise FileNotFoundError(
        f"⚠️ 请先下载Fama-French五因子数据到 {ff_data_path}!\n"
        "下载链接: http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html\n"
        "选择 'Fama/French 5 Factors (25 Portfolios)' → 'F-F_5_Research_Data_Factors'"
    )

# 读取因子数据（处理日期格式）
ff_df = pd.read_csv(ff_data_path, skiprows=3)  # 跳过前3行说明
ff_df = ff_df.rename(columns={'Unnamed: 0': 'Date'})  # 重命名日期列
ff_df['Date'] = pd.to_datetime(ff_df['Date'], format='%Y%m')  # 月度数据转为日期
ff_df = ff_df.set_index('Date').sort_index()  # 按日期排序

# === 3. 获取股票收益率数据（以SPY为例，代表市场指数） ===
print("\n🔍 正在下载SPY股票月度数据（10年历史）...")
spy = yf.download('SPY', period='10y', interval='1mo', progress=False)
spy = spy[['Close']]  # 只保留收盘价
spy = spy.resample('M').last()  # 月度收盘价（取月末）

# 计算月度收益率（超额收益率，假设无风险利率=0，实际需替换为真实无风险利率）
spy['Return'] = spy['Close'].pct_change()  # 月度简单收益率
spy = spy.dropna()  # 清理缺失值

# 重命名列方便合并
spy = spy.rename(columns={'Return': 'SPY_Return'})

# === 4. 合并因子数据与股票数据（按日期对齐） ===
combined = ff_df.join(spy, how='inner')  # 内连接，只保留共同日期
combined = combined.dropna()  # 清理缺失值

print(f"\n✅ 数据合并完成！共 {len(combined)} 个月度数据点")
print("样本日期范围:", combined.index.min().strftime('%Y-%m'), "至", combined.index.max().strftime('%Y-%m'))

# === 5. 构建五因子模型：SPY_Return = α + β1*MKT-RF + β2*SMB + β3*HML + β4*RMW + β5*CMA + ε ===
# 注意：因子数据中已包含MKT-RF（市场风险溢价），SPY_Return假设为超额收益率
X = combined[['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']]  # 因子数据
y = combined['SPY_Return']  # 目标变量（SPY月度超额收益率）

# 添加常数项（截距项α）
X = sm.add_constant(X)

# 拟合线性回归模型
model = sm.OLS(y, X).fit()

# === 6. 输出模型结果（关键指标） ===
print("\n📊 五因子模型回归结果：")
print(model.summary())

# === 7. 可视化因子贡献（可选） ===
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.bar(model.params.index[1:], model.params[1:], color='skyblue')
plt.axhline(y=0, color='r', linestyle='-')
plt.title('因子对SPY收益率的贡献系数', fontsize=14)
plt.xlabel('因子', fontsize=12)
plt.ylabel('回归系数', fontsize=12)
plt.tight_layout()
plt.savefig('factor_contributions.png')
print("\n✅ 因子贡献图已保存为 'factor_contributions.png'")

# === 8. 模型验证：计算R²和预测值 ===
r2 = model.rsquared
print(f"\n✅ 模型拟合优度 R²: {r2:.4f} (越接近1说明模型解释力越强)")

# 预测值（用于验证）
combined['Predicted'] = model.predict(X)
print("\n✅ 预测值已计算，示例：")
print(combined[['SPY_Return', 'Predicted']].head().round(4))