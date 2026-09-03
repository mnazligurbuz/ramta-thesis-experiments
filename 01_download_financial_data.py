# -*- coding: utf-8 -*-
"""
Step 1: Download financial market data (2015-2025).
Tickers: EUR/USD, GBP/USD, USD/JPY exchange rates + VIX index.
Computes log returns and rolling volatility. Saves to CSV.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import yfinance as yf

START, END = '2015-01-01', '2025-06-30'
TICKERS = {
    'EURUSD': 'EURUSD=X',
    'GBPUSD': 'GBPUSD=X',
    'USDJPY': 'USDJPY=X',
    'VIX':    '^VIX',
}

frames = {}
for name, tk in TICKERS.items():
    df = yf.download(tk, start=START, end=END, progress=False, auto_adjust=True)
    if df.empty:
        print(f'WARNING: no data for {name} ({tk})')
        continue
    # flatten possible MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    close = df['Close'].rename(name)
    frames[name] = close
    print(f'{name}: {len(close)} rows  ({close.index.min().date()} .. {close.index.max().date()})')

prices = pd.concat(frames.values(), axis=1).ffill().dropna()

# Log returns
returns = np.log(prices / prices.shift(1)).dropna()
returns.columns = [c + '_ret' for c in returns.columns]

# Rolling volatility (5, 22 day) for FX pairs
vol = pd.DataFrame(index=returns.index)
for c in ['EURUSD_ret', 'GBPUSD_ret', 'USDJPY_ret']:
    vol[c.replace('_ret', '_vol5')]  = returns[c].rolling(5).std()
    vol[c.replace('_ret', '_vol22')] = returns[c].rolling(22).std()

out = prices.join(returns).join(vol).dropna()
out.index.name = 'date'
out.to_csv(r'C:\Users\melik\OneDrive\Desktop\newthesis\experiments\data\financial_data.csv')
print(f'\nSaved financial_data.csv: {out.shape[0]} rows x {out.shape[1]} cols')
print(out.tail(3).round(5).to_string())
