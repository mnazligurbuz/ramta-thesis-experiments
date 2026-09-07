# -*- coding: utf-8 -*-
"""Build the Colab notebook for FinBERT + RAF + SHAP experiments.

Terminology note: this pipeline implements Retrieval-Augmented *Forecasting*
(RAF), not classical Retrieval-Augmented Generation (RAG). There is no
generative component: retrieval feeds a forecasting/classification head.
See thesis Section 2.6.3 for the distinction. The term RAG is used in this
repository only when referring to the classical generative literature."""
import json, io

cells = []
def md(src): cells.append({"cell_type":"markdown","metadata":{},"source":src.splitlines(keepends=True)})
def code(src): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":src.splitlines(keepends=True)})

md("""# RAMTA Experiments — FinBERT + RAF + SHAP (Text + RAF scope)
**Thesis:** Analysis of the Impact of Political Multimodal Discourse on Financial Markets — Melike Nazlı Gürbüz (s34853)

**How to use:**
1. Runtime → Change runtime type → **T4 GPU**
2. Upload `political_events.csv` when prompted (Cell 2)
3. Run all cells in order. Download the result files at the end.

Pipeline: FinBERT embeddings → (a) text-only classifier, (b) RAF-augmented classifier (FAISS retrieval over historical events) → SHAP explainability.

**Terminology:** this is Retrieval-Augmented **Forecasting** (RAF), not classical
Retrieval-Augmented Generation (RAG) — there is no generative component; retrieval
feeds a classification head (thesis Section 2.6.3).

**Retrieval integration (implementation vs. design):** the architectural
specification in thesis Section 5.6 integrates retrieved records through a
cross-attention layer. This text-scope pilot instead summarises the top-K
retrieved neighbours into four scalar context features that are concatenated to
the FinBERT embedding (see Cell 6). This is a deliberate simplification for the
current dataset size and is reported as such in thesis Section 7.6.""")

code("""!pip -q install transformers faiss-cpu shap scikit-learn
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE - enable GPU runtime!')""")

code("""# Cell 2 - upload the event dataset
from google.colab import files
up = files.upload()   # choose political_events.csv
import pandas as pd
ev = pd.read_csv('political_events.csv', parse_dates=['date'])
ev = ev.dropna(subset=['EURUSD_dir_3d']).sort_values('date').reset_index(drop=True)
# Binary task: UP vs DOWN (NEUTRAL dropped - too few samples for a 3-class task;
# at the final 202-event dataset scale this leaves 180 UP/DOWN events)
ev = ev[ev['EURUSD_dir_3d'].isin(['UP','DOWN'])].reset_index(drop=True)
ev['label'] = (ev['EURUSD_dir_3d'] == 'UP').astype(int)
print(len(ev), 'events |', ev['label'].value_counts().to_dict())""")

code("""# Cell 3 - FinBERT embeddings (frozen encoder -> CLS token)
# Note: with ~125 training events (final 202-event dataset), full fine-tuning of
# FinBERT's 110M parameters would overfit severely.
# A frozen domain-adapted encoder + light classifier head is the methodologically
# sound choice at this data scale (documented in thesis Section 4.2).
from transformers import AutoTokenizer, AutoModel
import numpy as np, torch

tok = AutoTokenizer.from_pretrained('ProsusAI/finbert')
enc = AutoModel.from_pretrained('ProsusAI/finbert').cuda().eval()

texts = (ev['actor'] + '. ' + ev['text']).tolist()
embs = []
with torch.no_grad():
    for i in range(0, len(texts), 16):
        batch = tok(texts[i:i+16], padding=True, truncation=True,
                    max_length=128, return_tensors='pt').to('cuda')
        out = enc(**batch)
        embs.append(out.last_hidden_state[:,0,:].cpu().numpy())  # CLS token
X = np.vstack(embs)
y = ev['label'].values
print('Embeddings:', X.shape)""")

code("""# Cell 4 - chronological split (70/15/15), same protocol as econometric baselines
n = len(ev); n_tr, n_va = int(n*0.70), int(n*0.15)
Xtr, ytr = X[:n_tr], y[:n_tr]
Xva, yva = X[n_tr:n_tr+n_va], y[n_tr:n_tr+n_va]
Xte, yte = X[n_tr+n_va:], y[n_tr+n_va:]
print(f'train={len(ytr)} val={len(yva)} test={len(yte)}')
print('test period:', ev.date.iloc[n_tr+n_va].date(), '..', ev.date.iloc[-1].date())""")

code("""# Cell 5 - Model A: FinBERT text-only classifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report

clf_text = LogisticRegression(max_iter=2000, C=0.1, class_weight='balanced')
clf_text.fit(Xtr, ytr)
pred_text = clf_text.predict(Xte)
acc_text = accuracy_score(yte, pred_text)
f1_text  = f1_score(yte, pred_text, average='weighted')
print(f'FinBERT text-only  |  test acc = {acc_text:.3f}   weighted F1 = {f1_text:.3f}')
print(classification_report(yte, pred_text, target_names=['DOWN','UP']))""")

code("""# Cell 6 - Model B: FinBERT + RAF (FAISS retrieval over historical events)
# For each event, retrieve top-K most similar events from the knowledge base and
# append their outcome statistics as four scalar context features.
#
# Leakage protocol: the FAISS knowledge base is built ONLY from the training
# split (index.add(Xn[:n_tr])), so no validation/test event can ever be
# retrieved, and every neighbour of a test event is chronologically prior to it.
# Within the training split a query may retrieve a train event that is later in
# time than itself; this affects only the fitting of the classifier head, never
# the reported test-set estimates (thesis Section 2.6.3).
#
# NOTE: K is fixed at 5 here. Thesis Section 5.6 lists K in {1,3,5,10} as the
# design-level search space; no K sweep was run at the current dataset size.
import faiss

K = 5
Xn = X / np.linalg.norm(X, axis=1, keepdims=True)   # cosine sim via normalized IP
index = faiss.IndexFlatIP(X.shape[1])
index.add(Xn[:n_tr].astype('float32'))               # KB = train events only

def raf_features(split_X, offset):
    feats = []
    for i, v in enumerate(split_X):
        q = (v / np.linalg.norm(v)).astype('float32')[None, :]
        sims, idxs = index.search(q, min(K + 1, n_tr))
        pairs = [(s, j) for s, j in zip(sims[0], idxs[0]) if (offset + i) != j][:K]
        lbls = np.array([ytr[j] for _, j in pairs], dtype=float)
        sim_arr = np.array([s for s, _ in pairs], dtype=float)
        feats.append([
            lbls.mean(),                                          # share of UP retrieved
            (sim_arr * lbls).sum() / max(sim_arr.sum(), 1e-8),    # similarity-weighted vote
            sim_arr.mean(),                                       # avg similarity (confidence)
            lbls.std() if len(lbls) > 1 else 0.0,                 # retrieval disagreement
        ])
    return np.array(feats)

Xtr_raf = np.hstack([Xtr, raf_features(Xtr, 0)])
Xte_raf = np.hstack([Xte, raf_features(Xte, n_tr + n_va)])

clf_raf = LogisticRegression(max_iter=2000, C=0.1, class_weight='balanced')
clf_raf.fit(Xtr_raf, ytr)
pred_raf = clf_raf.predict(Xte_raf)
acc_raf = accuracy_score(yte, pred_raf)
f1_raf  = f1_score(yte, pred_raf, average='weighted')
print(f'FinBERT + RAF (K={K})  |  test acc = {acc_raf:.3f}   weighted F1 = {f1_raf:.3f}')
print(classification_report(yte, pred_raf, target_names=['DOWN','UP']))""")

code("""# Cell 7 - Retrieval inspection: which historical events does RAF surface?
print('Example retrievals for test events:')
for i in range(min(3, len(Xte))):
    q = (Xte[i] / np.linalg.norm(Xte[i])).astype('float32')[None, :]
    sims, idxs = index.search(q, 3)
    print()
    row = ev.iloc[n_tr + n_va + i]
    print('QUERY :', row['date'].date(), '-', row['text'][:90])
    for s, j in zip(sims[0], idxs[0]):
        print(f'  [{s:.3f}]', ev.iloc[j]['date'].date(), '-', ev.iloc[j]['text'][:80])""")

code("""# Cell 8 - SHAP explainability on the RAF model
import shap
import numpy as np

feature_names = [f'emb_{i}' for i in range(768)] + [
    'raf_up_share', 'raf_weighted_vote', 'raf_avg_sim', 'raf_disagreement']
explainer = shap.LinearExplainer(clf_raf, Xtr_raf)
shap_vals = explainer.shap_values(Xte_raf)

mean_abs = np.abs(shap_vals).mean(axis=0)
imp = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
print('Top-15 features by mean |SHAP|:')
print(imp.head(15).round(4).to_string())
print()
raf_total = imp[['raf_up_share','raf_weighted_vote','raf_avg_sim','raf_disagreement']].sum()
txt_total = imp.filter(like='emb_').sum()
print(f'Aggregate |SHAP| - text embedding dims:  {txt_total:.4f}')
print(f'Aggregate |SHAP| - RAF context features: {raf_total:.4f}')""")

code("""# Cell 9 - results table + downloads
# The econometric baselines are NOT recomputed here; they are produced locally by
# 03_econometric_baselines.py and read from its committed output so that this
# table can never drift from the values reported in the thesis (Table 7.2).
# At the final 202-event dataset scale that script yields:
#   Majority class = 0.484, ARIMA(1,0,1) = 0.516   (data/results_econometric.csv)
# Note these two baselines are evaluated on the 3-class (UP/DOWN/NEUTRAL) sample
# of 199 labelled events (139/29/31 split), whereas the FinBERT rows below are
# evaluated on the 180 UP/DOWN events only (125/27/28 split). The two blocks are
# comparable in scale but are not computed on identical samples - see the
# footnote to Table 7.2 in the thesis.
import pandas as pd
MAJORITY_ACC, ARIMA_ACC = 0.484, 0.516   # from data/results_econometric.csv
res = pd.DataFrame([
    ['Majority class (3-class, n_test=31)', f'{MAJORITY_ACC:.3f}', '-'],
    ['ARIMA(1,0,1) (3-class, n_test=31)',   f'{ARIMA_ACC:.3f}',    '-'],
    ['FinBERT text-only (binary, n_test=28)',      f'{acc_text:.3f}', f'{f1_text:.3f}'],
    [f'FinBERT + RAF (K={K}) (binary, n_test=28)', f'{acc_raf:.3f}',  f'{f1_raf:.3f}'],
], columns=['Model', 'Directional Accuracy (test)', 'Weighted F1'])
print(res.to_string(index=False))
res.to_csv('results_finbert_raf.csv', index=False)

# SHAP summary plot for the thesis
import matplotlib.pyplot as plt
shap.summary_plot(shap_vals[:, -4:], Xte_raf[:, -4:],
                  feature_names=['RAF: share of UP', 'RAF: weighted vote',
                                 'RAF: avg similarity', 'RAF: disagreement'],
                  show=False)
plt.tight_layout(); plt.savefig('fig_shap_raf_features.png', dpi=200, bbox_inches='tight')
print('saved fig_shap_raf_features.png')

from google.colab import files
files.download('results_finbert_raf.csv')
files.download('fig_shap_raf_features.png')""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

out = r'C:\Users\melik\OneDrive\Desktop\newthesis\experiments\RAMTA_experiments_colab.ipynb'
with io.open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('notebook written:', out)
