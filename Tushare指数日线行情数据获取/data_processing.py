import pandas as pd
import os

def calculate_returns(input_filename, output_filename=None):
    """
    读取指数日线行情数据Excel文件，计算日收益率、隔夜收益率和日内收益率，并输出为新的Excel文件。
    
    由于开盘时交易，我们需要预测涨跌，所以信息不能超过今日开盘

    日收益率 = 当日开盘价 / 前一日开盘价 - 1
    隔夜收益率 = 当日开盘价 / 昨日收盘价 - 1
    日内收益率 = 昨日收盘价 / 昨日开盘价 - 1

    :param input_filename: 原始数据表格文件名
    :param output_filename: 输出新表格文件名，如未指定则自动生成
    """
    # 读取Excel文件
    df = pd.read_excel(input_filename)
    
    # 检查所需列是否存在
    required_columns = {'open', 'close'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"数据文件中缺少必须的列: {required_columns - set(df.columns)}")

    # 计算所需收益率
    df['daily_return'] = df['close'].pct_change()
    df['overnight_return'] = df['open'] / df['close'].shift(1) - 1
    df['intraday_return'] = df['close'] / df['open'] - 1

    # 输出为新的Excel文件
    if output_filename is None:
        # 自动根据输入文件命名
        base, ext = os.path.splitext(input_filename)
        output_filename = base + "_with_returns.xlsx"
    
    df.to_excel(output_filename, index=False)
    print(f"已生成包含收益率的新表格: {output_filename}")


