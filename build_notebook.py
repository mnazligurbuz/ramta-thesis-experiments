# -*- coding: utf-8 -*-
"""Build the Colab notebook for FinBERT + RAG + SHAP experiments."""
import json, io

cells = []
def md(src): cells.append({"cell_type":"markdown","metadata":{},"source":src.splitlines(keepends=True)})
def code(src): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":src.splitlines(keepends=True)})

md("""# RAMTA Experiments — FinBERT + RAG + SHAP (Text + RAG scope)
**Thesis:** Analysis of the Impact of Political Multimodal Discourse on Financial Markets — Melike Nazlı Gürbüz (s34853)

**How to use:**
1. Runtime → Change runtime type → **T4 GPU**
2. Upload `political_events.csv` when prompted (Cell 2)
3. Run all cells in order. Download the result files at the end.

Pipeline: FinBERT embeddings → (a) text-only classifier, (b) RAG-augmented classifier (FAISS retrieval over historical events) → SHAP explainability.""")

code("""!pip -q install transformers faiss-cpu shap scikit-learn
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE - enable GPU runtime!')""")

code("""# Cell 2 - upload the event dataset
from google.colab import files
up = files.upload()   # choose political_events.csv
import pandas as pd
ev = pd.read_csv('political_events.csv', parse_dates=['date'])
ev = ev.dropna(subset=['EURUSD_dir_3d']).sort_values('date').reset_index(drop=True)
# Binary task: UP vs DOWN (NEUTRAL dropped - too few samples for 3-class at n~78)
ev = ev[ev['EURUSD_dir_3d'].isin(['UP','DOWN'])].reset_index(drop=True)
ev['label'] = (ev['EURUSD_dir_3d'] == 'UP').astype(int)
print(len(ev), 'events |', ev['label'].value_counts().to_dict())""")

code("""# Cell 3 - FinBERT embeddings (frozen encoder -> CLS token)
# Note: with ~50 training events, full fine-tuning would overfit severely.
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

code("""# Cell 6 - Model B: FinBERT + RAG (FAISS retrieval over historical events)
# For each event, retrieve top-K most similar PAST events (train set only -
# strictly no leakage) and append their outcome statistics as context features.
import faiss

K = 5
Xn = X / np.linalg.norm(X, axis=1, keepdims=True)   # cosine sim via normalized IP
index = faiss.IndexFlatIP(X.shape[1])
index.add(Xn[:n_tr].astype('float32'))               # KB = train events only

def rag_features(split_X, offset):
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

Xtr_rag = np.hstack([Xtr, rag_features(Xtr, 0)])
Xte_rag = np.hstack([Xte, rag_features(Xte, n_tr + n_va)])

clf_rag = LogisticRegression(max_iter=2000, C=0.1, class_weight='balanced')
clf_rag.fit(Xtr_rag, ytr)
pred_rag = clf_rag.predict(Xte_rag)
acc_rag = accuracy_score(yte, pred_rag)
f1_rag  = f1_score(yte, pred_rag, average='weighted')
print(f'FinBERT + RAG (K={K})  |  test acc = {acc_rag:.3f}   weighted F1 = {f1_rag:.3f}')
print(classification_report(yte, pred_rag, target_names=['DOWN','UP']))""")

code("""# Cell 7 - Retrieval inspection: which historical events does RAG surface?
print('Example retrievals for test events:')
for i in range(min(3, len(Xte))):
    q = (Xte[i] / np.linalg.norm(Xte[i])).astype('float32')[None, :]
    sims, idxs = index.search(q, 3)
    print()
    row = ev.iloc[n_tr + n_va + i]
    print('QUERY :', row['date'].date(), '-', row['text'][:90])
    for s, j in zip(sims[0], idxs[0]):
        print(f'  [{s:.3f}]', ev.iloc[j]['date'].date(), '-', ev.iloc[j]['text'][:80])""")

code("""# Cell 8 - SHAP explainability on the RAG model
import shap
import numpy as np

feature_names = [f'emb_{i}' for i in range(768)] + [
    'rag_up_share', 'rag_weighted_vote', 'rag_avg_sim', 'rag_disagreement']
explainer = shap.LinearExplainer(clf_rag, Xtr_rag)
shap_vals = explainer.shap_values(Xte_rag)

mean_abs = np.abs(shap_vals).mean(axis=0)
imp = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
print('Top-15 features by mean |SHAP|:')
print(imp.head(15).round(4).to_string())
print()
rag_total = imp[['rag_up_share','rag_weighted_vote','rag_avg_sim','rag_disagreement']].sum()
txt_total = imp.filter(like='emb_').sum()
print(f'Aggregate |SHAP| - text embedding dims:  {txt_total:.4f}')
print(f'Aggregate |SHAP| - RAG context features: {rag_total:.4f}')""")

code("""# Cell 9 - results table + downloads
import pandas as pd
res = pd.DataFrame([
    ['Majority class',         '0.385 (local run)', '-'],
    ['ARIMA(1,0,1)',           '0.385 (local run)', '-'],
    ['FinBERT text-only',      f'{acc_text:.3f}', f'{f1_text:.3f}'],
    [f'FinBERT + RAG (K={K})', f'{acc_rag:.3f}', f'{f1_rag:.3f}'],
], columns=['Model', 'Directional Accuracy (test)', 'Weighted F1'])
print(res.to_string(index=False))
res.to_csv('results_finbert_rag.csv', index=False)

# SHAP summary plot for the thesis
import matplotlib.pyplot as plt
shap.summary_plot(shap_vals[:, -4:], Xte_rag[:, -4:],
                  feature_names=['RAG: share of UP', 'RAG: weighted vote',
                                 'RAG: avg similarity', 'RAG: disagreement'],
                  show=False)
plt.tight_layout(); plt.savefig('fig_shap_rag_features.png', dpi=200, bbox_inches='tight')
print('saved fig_shap_rag_features.png')

from google.colab import files
files.download('results_finbert_rag.csv')
files.download('fig_shap_rag_features.png')""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

out = r'C:\Users\melik\OneDrive\Desktop\newthesis\experiments\RAMTA_experiments_colab.ipynb'
with io.open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('notebook written:', out)
