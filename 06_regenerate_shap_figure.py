# -*- coding: utf-8 -*-
"""
Step 6: Reproduce the Text + RAF pilot locally (CPU) and regenerate Figure 7.2.

This mirrors RAMTA_experiments_colab.ipynb exactly - same frozen FinBERT CLS
embeddings, same chronological 70/15/15 split, same four RAF context features,
same regularised logistic head, same SHAP LinearExplainer - so that the figure
used in the thesis is produced by committed code rather than by an ad-hoc
notebook run, and so that the accuracies reported in Table 7.2 can be verified.

Output: fig_shap_raf_features.png (Figure 7.2) + printed accuracies.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import faiss
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = r'C:\Users\melik\OneDrive\Desktop\newthesis'
DATA = ROOT + r'\experiments\data'

# ── Data ──────────────────────────────────────────────────────────────────────
ev = pd.read_csv(f'{DATA}\\political_events.csv', parse_dates=['date'])
ev = ev.dropna(subset=['EURUSD_dir_3d']).sort_values('date').reset_index(drop=True)
ev = ev[ev['EURUSD_dir_3d'].isin(['UP', 'DOWN'])].reset_index(drop=True)
ev['label'] = (ev['EURUSD_dir_3d'] == 'UP').astype(int)
print(f'{len(ev)} UP/DOWN events | {ev["label"].value_counts().to_dict()}')

# ── Frozen FinBERT CLS embeddings ─────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained('ProsusAI/finbert')
enc = AutoModel.from_pretrained('ProsusAI/finbert').eval()

texts = (ev['actor'] + '. ' + ev['text']).tolist()
embs = []
with torch.no_grad():
    for i in range(0, len(texts), 16):
        batch = tok(texts[i:i+16], padding=True, truncation=True,
                    max_length=128, return_tensors='pt')
        embs.append(enc(**batch).last_hidden_state[:, 0, :].numpy())
X = np.vstack(embs)
y = ev['label'].values
print('Embeddings:', X.shape)

# ── Chronological split ───────────────────────────────────────────────────────
n = len(ev); n_tr, n_va = int(n*0.70), int(n*0.15)
Xtr, ytr = X[:n_tr], y[:n_tr]
Xte, yte = X[n_tr+n_va:], y[n_tr+n_va:]
print(f'train={len(ytr)} val={n_va} test={len(yte)}')

# ── Model A: text-only ────────────────────────────────────────────────────────
clf_text = LogisticRegression(max_iter=2000, C=0.1, class_weight='balanced').fit(Xtr, ytr)
pred_text = clf_text.predict(Xte)
acc_text, f1_text = accuracy_score(yte, pred_text), f1_score(yte, pred_text, average='weighted')
print(f'FinBERT text-only  |  acc = {acc_text:.3f}  F1 = {f1_text:.3f}')

# ── Model B: + RAF (FAISS over train-only knowledge base) ─────────────────────
K = 5
Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
index = faiss.IndexFlatIP(X.shape[1])
index.add(Xn[:n_tr].astype('float32'))

def raf_features(split_X, offset):
    feats = []
    for i, v in enumerate(split_X):
        q = (v / np.linalg.norm(v)).astype('float32')[None, :]
        sims, idxs = index.search(q, min(K + 1, n_tr))
        pairs = [(s, j) for s, j in zip(sims[0], idxs[0]) if (offset + i) != j][:K]
        lbls = np.array([ytr[j] for _, j in pairs], dtype=float)
        sim_arr = np.array([s for s, _ in pairs], dtype=float)
        feats.append([
            lbls.mean(),
            (sim_arr * lbls).sum() / max(sim_arr.sum(), 1e-8),
            sim_arr.mean(),
            lbls.std() if len(lbls) > 1 else 0.0,
        ])
    return np.array(feats)

Xtr_raf = np.hstack([Xtr, raf_features(Xtr, 0)])
Xte_raf = np.hstack([Xte, raf_features(Xte, n_tr + n_va)])

clf_raf = LogisticRegression(max_iter=2000, C=0.1, class_weight='balanced').fit(Xtr_raf, ytr)
pred_raf = clf_raf.predict(Xte_raf)
acc_raf, f1_raf = accuracy_score(yte, pred_raf), f1_score(yte, pred_raf, average='weighted')
print(f'FinBERT + RAF (K={K})  |  acc = {acc_raf:.3f}  F1 = {f1_raf:.3f}')

# ── SHAP on the four RAF context features -> Figure 7.2 ───────────────────────
explainer = shap.LinearExplainer(clf_raf, Xtr_raf)
shap_vals = explainer.shap_values(Xte_raf)

plt.figure()
shap.summary_plot(shap_vals[:, -4:], Xte_raf[:, -4:],
                  feature_names=['RAF: share of UP', 'RAF: weighted vote',
                                 'RAF: avg similarity', 'RAF: disagreement'],
                  show=False)
plt.tight_layout()
plt.savefig(f'{ROOT}\\fig_shap_raf_features.png', dpi=200, bbox_inches='tight')
print(f'\nSaved figure: {ROOT}\\fig_shap_raf_features.png')

pd.DataFrame([
    ['Majority class (3-class, n_test=31)', '0.484', '-'],
    ['ARIMA(1,0,1) (3-class, n_test=31)',   '0.516', '-'],
    [f'FinBERT text-only (binary, n_test={len(yte)})', f'{acc_text:.3f}', f'{f1_text:.3f}'],
    [f'FinBERT + RAF (K={K}) (binary, n_test={len(yte)})', f'{acc_raf:.3f}', f'{f1_raf:.3f}'],
], columns=['Model', 'Directional Accuracy (test)', 'Weighted F1']
).to_csv(f'{DATA}\\results_finbert_raf.csv', index=False)
print('Saved: results_finbert_raf.csv')
