# -*- coding: utf-8 -*-
"""
Sensitivity analysis for the 0.1% neutral-band classification threshold
(supervisor comment, Section 3.2.1 / 4.4). Re-labels the 202-event dataset
at alternative thresholds (0.05%, 0.1%, 0.2%, 0.3%) and reports how the
label distribution and majority-class baseline change, to assess whether
the reported pilot results are sensitive to this specific choice.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import pandas as pd

DATA = r'C:\Users\melik\OneDrive\Desktop\newthesis\experiments\data'
fin = pd.read_csv(f'{DATA}\\financial_data.csv', parse_dates=['date'], index_col='date')
ev = pd.read_csv(f'{DATA}\\political_events.csv', parse_dates=['date'])

def forward_return(series_price, event_date, horizon_days):
    idx = series_price.index
    base_pos = idx.searchsorted(event_date, side='right') - 1
    if base_pos < 0:
        return np.nan
    tgt_date = event_date + pd.Timedelta(days=horizon_days)
    tgt_pos = idx.searchsorted(tgt_date, side='right') - 1
    if tgt_pos <= base_pos or tgt_pos >= len(idx):
        return np.nan
    return float(np.log(series_price.iloc[tgt_pos] / series_price.iloc[base_pos]))

# Recompute raw 3-day EUR/USD returns for all events (same as original label pipeline)
ev['EURUSD_ret_3d_raw'] = ev['date'].apply(lambda d: forward_return(fin['EURUSD'], d, 3))
ev = ev.dropna(subset=['EURUSD_ret_3d_raw']).sort_values('date').reset_index(drop=True)

n = len(ev)
n_tr, n_va = int(n*0.70), int(n*0.15)
test_ev = ev.iloc[n_tr+n_va:]

print(f'Total events with valid returns: {n}')
print(f'Test-set size: {len(test_ev)}')
print()

results = []
for thresh_pct in [0.05, 0.10, 0.20, 0.30]:
    thresh = thresh_pct / 100
    def direction(r):
        if r > thresh: return 'UP'
        if r < -thresh: return 'DOWN'
        return 'NEUTRAL'
    ev['dir'] = ev['EURUSD_ret_3d_raw'].apply(direction)
    test_dir = ev.iloc[n_tr+n_va:]['dir']

    dist = test_dir.value_counts(normalize=True) * 100
    majority_label = ev.iloc[:n_tr]['dir'].mode()[0]  # majority computed on TRAIN only
    maj_acc = (test_dir == majority_label).mean()

    # binary UP/DOWN majority-class accuracy (excluding NEUTRAL), matching FinBERT pilot protocol
    bin_test = test_dir[test_dir.isin(['UP','DOWN'])]
    bin_train_mode = ev.iloc[:n_tr]['dir']
    bin_train_mode = bin_train_mode[bin_train_mode.isin(['UP','DOWN'])].mode()[0]
    bin_maj_acc = (bin_test == bin_train_mode).mean() if len(bin_test) else float('nan')

    results.append({
        'threshold_%': thresh_pct,
        'test_UP_%': round(dist.get('UP', 0), 1),
        'test_DOWN_%': round(dist.get('DOWN', 0), 1),
        'test_NEUTRAL_%': round(dist.get('NEUTRAL', 0), 1),
        'n_test_UP_DOWN_only': len(bin_test),
        'majority_class_acc_3way': round(maj_acc, 3),
        'majority_class_acc_binary': round(bin_maj_acc, 3),
    })

res_df = pd.DataFrame(results)
print(res_df.to_string(index=False))
res_df.to_csv(f'{DATA}\\results_threshold_sensitivity.csv', index=False)
print()
print(f'Saved: {DATA}\\results_threshold_sensitivity.csv')
