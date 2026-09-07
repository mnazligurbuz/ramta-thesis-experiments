# -*- coding: utf-8 -*-
"""
Step 5: Descriptive daily-return statistics supporting the 0.1% neutral-band
justification reported in thesis Section 3.2.1.

For each FX pair this reports:
  - the share of trading days whose daily log return falls inside the +/-0.1%
    neutral band (i.e. would be labelled NEUTRAL by the labelling rule in
    02_build_event_dataset.py), and
  - the unconditional standard deviation of daily log returns.

The point of the comparison is that a single common threshold produces a
similar NEUTRAL coverage across pairs of differing volatility, which is why a
common band is used rather than pair-specific ones.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd

DATA = r'C:\Users\melik\OneDrive\Desktop\newthesis\experiments\data'
THRESH = 0.001  # 0.1% - identical to THRESH in 02_build_event_dataset.py

fin = pd.read_csv(f'{DATA}\\financial_data.csv', parse_dates=['date'], index_col='date')

rows = []
for pair in ['EURUSD', 'GBPUSD', 'USDJPY']:
    r = fin[f'{pair}_ret'].dropna()
    rows.append({
        'pair': pair,
        'n_days': len(r),
        'neutral_share_%': round(float((r.abs() <= THRESH).mean()) * 100, 1),
        'daily_std_%': round(float(r.std()) * 100, 2),
    })

out = pd.DataFrame(rows)
print(f'Neutral band: +/-{THRESH*100:.1f}%   period: '
      f'{fin.index.min().date()} .. {fin.index.max().date()}')
print()
print(out.to_string(index=False))

out.to_csv(f'{DATA}\\results_return_statistics.csv', index=False)
print(f'\nSaved: results_return_statistics.csv')
