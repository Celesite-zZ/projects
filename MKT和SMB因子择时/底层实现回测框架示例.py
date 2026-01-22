# -*- coding: utf-8 -*-
"""
Created on Sun Dec 28 11:35:24 2025

@author: zhangziyu
"""

import pandas as pd
import numpy as np

class SimpleDataLoader:
    """最简单的数据加载器"""
    
    def __init__(self):
        self.prices = None  # 存储价格数据
        
    def load_from_csv(self, filepath):
        """从CSV加载数据（只需要日期和收盘价）"""
        # CSV格式示例：
        # date,open,high,low,close,volume
        # 2020-01-01,100,102,99,101,10000
        
        df = pd.read_csv(filepath)
        df['date'] = pd.to_datetime(df['date'])  # 确保日期是时间格式
        df = df[['date', 'close']]  # 我们只需要日期和收盘价
        df.set_index('date', inplace=True)
        
        self.prices = df
        return df
    
    def get_price(self, date):
        """获取某一天的收盘价"""
        if date in self.prices.index:
            return self.prices.loc[date, 'close']
        return None
    
    
class SimpleStrategy:
    """简单策略：金叉死叉策略"""
    
    def __init__(self, short_window=5, long_window=20):
        self.short_window = short_window  # 短期均线天数
        self.long_window = long_window    # 长期均线天数
        self.position = 0  # 持仓状态：0空仓，1持仓
        
    def calculate_signals(self, prices, current_index):
        """计算交易信号"""
        if current_index < self.long_window:
            return "hold"  # 数据不足，不交易
            
        # 计算移动平均线
        short_ma = prices['close'][current_index-self.short_window:current_index].mean()
        long_ma = prices['close'][current_index-self.long_window:current_index].mean()
        
        current_price = prices['close'].iloc[current_index]
        
        # 金叉：短期均线上穿长期均线，买入
        if short_ma > long_ma and self.position == 0:
            self.position = 1  # 标记为持仓状态
            return "buy"
        
        # 死叉：短期均线下穿长期均线，卖出
        elif short_ma < long_ma and self.position == 1:
            self.position = 0  # 标记为空仓状态
            return "sell"
        
        return "hold"    
    
    
class SimpleBacktester:
    """最简单的回测引擎"""
    
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital  # 初始资金
        self.cash = initial_capital              # 可用现金
        self.holdings = 0                        # 持仓数量
        self.current_date = None                 # 当前日期
        self.current_price = 0                   # 当前价格
        
        # 记录回测结果
        self.history = []  # 记录每日状态
        self.trades = []   # 记录交易记录
        
    def run_backtest(self, prices, strategy):
        """运行回测主循环"""
        
        print(f"开始回测，初始资金：{self.initial_capital}")
        print(f"数据天数：{len(prices)}")
        
        # 逐日回测
        for i in range(len(prices)):
            # 获取当前日期和价格
            current_date = prices.index[i]
            current_price = prices['close'].iloc[i]
            
            self.current_date = current_date
            self.current_price = current_price
            
            # 计算策略信号
            signal = strategy.calculate_signals(prices, i)
            
            # 执行交易
            if signal == "buy" and self.cash > 0:
                # 计算能买多少股（简单全仓买入）
                can_buy = self.cash // current_price
                if can_buy > 0:
                    cost = can_buy * current_price
                    self.holdings += can_buy
                    self.cash -= cost
                    
                    # 记录交易
                    self.trades.append({
                        'date': current_date,
                        'type': 'buy',
                        'price': current_price,
                        'shares': can_buy,
                        'cash_after': self.cash
                    })
                    
                    print(f"{current_date.date()}: 买入 {can_buy}股，价格{current_price:.2f}")
                    
            elif signal == "sell" and self.holdings > 0:
                # 卖出所有持仓
                value = self.holdings * current_price
                self.cash += value
                
                # 记录交易
                self.trades.append({
                    'date': current_date,
                    'type': 'sell',
                    'price': current_price,
                    'shares': self.holdings,
                    'cash_after': self.cash
                })
                
                print(f"{current_date.date()}: 卖出 {self.holdings}股，价格{current_price:.2f}")
                self.holdings = 0
            
            # 计算当日资产总值
            portfolio_value = self.cash + (self.holdings * current_price)
            
            # 记录每日状态
            self.history.append({
                'date': current_date,
                'price': current_price,
                'holdings': self.holdings,
                'cash': self.cash,
                'total_value': portfolio_value,
                'signal': signal
            })
        
        # 回测结束，打印结果
        self.print_results()
        
    def print_results(self):
        """打印回测结果"""
        if not self.history:
            print("没有回测数据！")
            return
            
        final_value = self.history[-1]['total_value']
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        print("\n" + "="*50)
        print("回测结果总结")
        print("="*50)
        print(f"初始资金: ¥{self.initial_capital:,.2f}")
        print(f"最终资产: ¥{final_value:,.2f}")
        print(f"总收益率: {total_return:.2%}")
        print(f"总交易次数: {len(self.trades)}")
        print(f"最终持仓: {self.holdings}股")
        print(f"剩余现金: ¥{self.cash:,.2f}")
        
        # 计算最大回撤（简化版）
        values = [day['total_value'] for day in self.history]
        max_value = max(values)
        min_value_after_max = min(values[values.index(max_value):]) if max_value in values else min(values)
        max_drawdown = (max_value - min_value_after_max) / max_value if max_value > 0 else 0
        
        print(f"最大回撤: {max_drawdown:.2%}")
        print("="*50)
        
    def plot_results(self):
        """绘制简单的资金曲线"""
        try:
            import matplotlib.pyplot as plt
            
            dates = [day['date'] for day in self.history]
            values = [day['total_value'] for day in self.history]
            prices = [day['price'] for day in self.history]
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # 绘制资产曲线
            ax1.plot(dates, values, 'b-', linewidth=2, label='资产总值')
            ax1.axhline(y=self.initial_capital, color='r', linestyle='--', alpha=0.5, label='初始资金')
            ax1.set_title('资金曲线')
            ax1.set_ylabel('资产（元）')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 绘制价格曲线和买卖点
            ax2.plot(dates, prices, 'k-', linewidth=1, label='价格')
            
            # 标记买卖点
            buy_dates = [t['date'] for t in self.trades if t['type'] == 'buy']
            buy_prices = [t['price'] for t in self.trades if t['type'] == 'buy']
            sell_dates = [t['date'] for t in self.trades if t['type'] == 'sell']
            sell_prices = [t['price'] for t in self.trades if t['type'] == 'sell']
            
            ax2.scatter(buy_dates, buy_prices, color='g', s=100, marker='^', label='买入')
            ax2.scatter(sell_dates, sell_prices, color='r', s=100, marker='v', label='卖出')
            
            ax2.set_title('价格与交易信号')
            ax2.set_ylabel('价格（元）')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("请先安装matplotlib：pip install matplotlib")    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    