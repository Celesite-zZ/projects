# -*- coding: utf-8 -*-
"""
Created on Wed Dec 24 10:15:32 2025

@author: zhangziyu
"""
import tushare as ts
import pandas as pd
import os

file_name = '文件名未获取' #定义一个全局文件名变量方便引用

def read_token_from_txt(token_file="tushare_token.txt"):
    """
    从与本脚本同目录下的txt文件读取Tushare token。
    文件名默认为 'tushare_token.txt'。
    文件内容只需写一行token字符串，不要有多余空格。
    """
    token_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), token_file)
    if not os.path.isfile(token_path):
        raise FileNotFoundError(f"未找到{token_file}，请在脚本所在文件夹下放置并写入你的Tushare token。")
    with open(token_path, "r", encoding="utf-8") as f:
        token = f.readline().strip()
    if not token:
        raise ValueError(f"{token_file} 文件内容为空，请填入有效的Tushare token。")
    return token

def fetch_and_save_index_daily(ts_code:str = "000300.SH", start_date:str = "20230101", end_date:str = "20231231", excel_name:str = None, tushare_token:str = None, token_file:str = "tushare_token.txt"):
    """
    从Tushare导出指定指数日线行情数据，并保存为Excel。

    :param ts_code: 指数代码, 如'000300.SH'
    :param start_date: 开始日期, 格式YYYYMMDD
    :param end_date: 结束日期, 格式YYYYMMDD
    :param excel_name: 可选，Excel文件名(含扩展名)，若无则默认"指数日线_{ts_code}.xlsx"
    :param tushare_token: 可选，Tushare的token。若未提供则尝试用环境变量TUSHARE_TOKEN或TXT文件
    :param token_file: 储存token的txt文件名，默认"tushare_token.txt"
    """
    # 设置Token
    if not tushare_token:
        # 尝试环境变量
        tushare_token = os.environ.get("TUSHARE_TOKEN", None)
        if not tushare_token:
            # 尝试从txt文件读取
            tushare_token = read_token_from_txt(token_file=token_file)
    ts.set_token(tushare_token)
    pro = ts.pro_api()

    try:
        print(f"正在获取指数{ts_code}从{start_date}到{end_date}的日线行情数据...")
        df = pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    except Exception as e:
        print("获取数据失败：", e)
        return

    if df.empty:
        print("没有获取到数据，请检查参数设置。")
        return

    # 数据排序
    df.sort_values('trade_date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 生成导出Excel文件名
    if not excel_name:
        excel_name = f"指数日线_{ts_code.replace('.', '_')}_{start_date}_{end_date}.xlsx"
        
        global file_name 
        file_name = excel_name

    try:
        df.to_excel(excel_name, index=False)
        print(f"数据已成功导出至: {excel_name}")
    except Exception as e:
        print("导出数据至Excel失败：", e)
        
def fetch_and_save_fut_daily(ts_code:str = "000300.SH", start_date:str = "20230101", end_date:str = "20231231", excel_name:str = None, tushare_token:str = None, token_file:str = "tushare_token.txt"):
    """
    从Tushare导出指定期货日线行情数据，并保存为Excel。

    :param ts_code: 期货代码, 如'000300.SH'
    :param start_date: 开始日期, 格式YYYYMMDD
    :param end_date: 结束日期, 格式YYYYMMDD
    :param excel_name: 可选，Excel文件名(含扩展名)，若无则默认"期货日线_{ts_code}.xlsx"
    :param tushare_token: 可选，Tushare的token。若未提供则尝试用环境变量TUSHARE_TOKEN或TXT文件
    :param token_file: 储存token的txt文件名，默认"tushare_token.txt"
    """
    # 设置Token
    if not tushare_token:
        # 尝试环境变量
        tushare_token = os.environ.get("TUSHARE_TOKEN", None)
        if not tushare_token:
            # 尝试从txt文件读取
            tushare_token = read_token_from_txt(token_file=token_file)
    ts.set_token(tushare_token)
    pro = ts.pro_api()

    try:
        print(f"正在获取期货{ts_code}从{start_date}到{end_date}的日线行情数据...")
        df = pro.fut_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)
    except Exception as e:
        print("获取数据失败：", e)
        return

    if df.empty:
        print("没有获取到数据，请检查参数设置。")
        return

    # 数据排序
    df.sort_values('trade_date', inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 生成导出Excel文件名
    if not excel_name:
        excel_name = f"期货日线_{ts_code.replace('.', '_')}_{start_date}_{end_date}.xlsx"
        
        global file_name 
        file_name = excel_name

    try:
        df.to_excel(excel_name, index=False)
        print(f"数据已成功导出至: {excel_name}")
    except Exception as e:
        print("导出数据至Excel失败：", e)        
        
        
        
