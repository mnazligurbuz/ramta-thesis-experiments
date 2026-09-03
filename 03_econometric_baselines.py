# -*- coding: utf-8 -*-
"""
Step 3: Econometric baselines on the SAME chronological split used later
for FinBERT (events sorted by date: 70% train / 15% val / 15% test).

- Directional baseline: ARIMA(1,0,1) rolling forecast of 3-day-ahead return
  sign at each test event date. Plus naive majority-class baseline.
- Volatility: GARCH(1,1) fitted on train-period returns, report persistence
  and out-of-sample RMSE vs realised 5-day vol on test period.
"""
import sys, warnings
sys.stdout.reconfigure(encoding='utf-8')
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from arch import arch_model

DATA = r'C:\Users\melik\OneDrive\Desktop\newthesis\experiments\data'
fin = pd.read_csv(f'{DATA}\\financial_data.csv', parse_dates=['date'], index_col='date')
ev  = pd.read_csv(f'{DATA}\\political_events.csv', parse_dates=['date'])
ev  = ev.dropna(subset=['EURUSD_dir_3d']).sort_values('date').reset_index(drop=True)

n = len(ev)
n_tr, n_va = int(n*0.70), int(n*0.15)
train_ev, val_ev, test_ev = ev[:n_tr], ev[n_tr:n_tr+n_va], ev[n_tr+n_va:]
print(f'Events split: train={len(train_ev)}  val={len(val_ev)}  test={len(test_ev)}')
print(f'Test period: {test_ev.date.min().date()} .. {test_ev.date.max().date()}\n')

results = {}

# ── Majority-class baseline ───────────────────────────────────────────────────
maj = train_ev['EURUSD_dir_3d'].mode()[0]
acc_maj = (test_ev['EURUSD_dir_3d'] == maj).mean()
results['Majority class'] = acc_maj
print(f'Majority-class baseline ({maj}): test directional acc = {acc_maj:.3f}')

# ── ARIMA rolling directional forecast ───────────────────────────────────────
ret = fin['EURUSD_ret'].dropna()
correct = 0; total = 0
for _, row in test_ev.iterrows():
    d = row['date']
    hist = ret[ret.index <= d]
    if len(hist) < 100:
        continue
    try:
        m = ARIMA(hist.iloc[-500:], order=(1,0,1)).fit()
        fc = m.forecast(steps=3).sum()          # cumulative 3-day forecast
        pred = 'UP' if fc > 0.001 else ('DOWN' if fc < -0.001 else 'NEUTRAL')
        # ARIMA nearly always predicts ~0 -> map neutral to majority for fairness
        if pred == 'NEUTRAL':
            pred = maj
        correct += int(pred == row['EURUSD_dir_3d']); total += 1
    except Exception:
        continue
acc_arima = correct / total if total else float('nan')
results['ARIMA(1,0,1)'] = acc_arima
print(f'ARIMA(1,0,1) rolling: test directional acc = {acc_arima:.3f}  (n={total})')

# ── GARCH(1,1) volatility ─────────────────────────────────────────────────────
split_date = test_ev['date'].min()
r_tr = ret[ret.index <  split_date] * 100    # scale to % for arch
r_te = ret[ret.index >= split_date] * 100
g = arch_model(r_tr, vol='Garch', p=1, q=1, mean='Constant').fit(disp='off')
a1, b1 = g.params['alpha[1]'], g.params['beta[1]']
print(f'\nGARCH(1,1) on EURUSD (train period):')
print(f'  alpha1 = {a1:.4f}   beta1 = {b1:.4f}   persistence = {a1+b1:.4f}')

# out-of-sample: rolling 1-step conditional vol vs realized |return|
fc_g = g.forecast(horizon=1, start=r_tr.index[-1], reindex=False)
# refit-free approximation: filter test data through fitted params
g_full = arch_model(pd.concat([r_tr, r_te]), vol='Garch', p=1, q=1, mean='Constant')
res_fixed = g_full.fix(g.params.values)
cond_vol_te = res_fixed.conditional_volatility[-len(r_te):]
realized = r_te.abs()
rmse_garch = float(np.sqrt(np.mean((cond_vol_te.values - realized.values)**2)))
print(f'  Out-of-sample RMSE (cond.vol vs |ret|, %): {rmse_garch:.4f}')

# naive vol baseline: unconditional train std
rmse_naive = float(np.sqrt(np.mean((r_tr.std() - realized.values)**2)))
print(f'  Naive (unconditional) RMSE:               {rmse_naive:.4f}')

# ── Save results ──────────────────────────────────────────────────────────────
out = pd.DataFrame([
    ['Majority class', 'EURUSD dir (3d)', f'{acc_maj:.3f}', '-'],
    ['ARIMA(1,0,1)',   'EURUSD dir (3d)', f'{acc_arima:.3f}', '-'],
    ['GARCH(1,1)',     'EURUSD volatility', '-', f'RMSE={rmse_garch:.4f}, persistence={a1+b1:.3f}'],
    ['Naive vol',      'EURUSD volatility', '-', f'RMSE={rmse_naive:.4f}'],
], columns=['Model', 'Task', 'Directional Acc.', 'Volatility'])
out.to_csv(f'{DATA}\\results_econometric.csv', index=False)
print(f'\nSaved: results_econometric.csv')
