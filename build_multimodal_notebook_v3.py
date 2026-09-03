# -*- coding: utf-8 -*-
"""Build the Colab notebook for the extended real multimodal (Text+Audio+Visual+RAF) feasibility pilot.
v3: extends v2 (true sequence-level cross-attention) from N=5 to N=15 real Federal Reserve
press-conference events, all sourced from federalreserve.gov, and adds a minimal
real train/test split with actual EUR/USD 3-day directional labels so a preliminary
(not just architectural-feasibility) quantitative result can be reported, while still
being explicit that N=15 remains far too small for a statistically powered evaluation."""
import json, io

cells = []
def md(src): cells.append({"cell_type":"markdown","metadata":{},"source":src.splitlines(keepends=True)})
def code(src): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":src.splitlines(keepends=True)})

md("""# RAMTA Multimodal Feasibility Pilot v3 (Text + Audio + Visual + RAF, N=15)
**Thesis:** Analysis of the Impact of Political Multimodal Discourse on Financial Markets — Melike Nazlı Gürbüz (s34853)

This notebook implements and runs the **full multimodal RAMTA architecture end-to-end**
on N=15 real political events (FOMC press conferences, 2015-2024), each with real
text, audio, and video sourced directly from federalreserve.gov's own public webcast
archive.

**v3 changes from v2:**
- Extends the event set from N=5 to **N=15** (still all real Federal Reserve press
  conferences, same sourcing method, same legal basis).
- Uses the same **true sequence-level cross-attention** as v2 (Section 5.5 formulation),
  not pooled-vector attention.
- Adds a minimal real **train/test split** (11 train / 4 test) using the actual EUR/USD
  3-day directional labels already computed for the Text+RAF pilot (Chapter 7,
  `political_events.csv`), so this notebook reports a genuine (if very small-sample)
  **directional-accuracy result** for the fused multimodal representation, not only an
  architecture/feasibility demonstration.

**Important limitation, stated explicitly:** N=15 (11 train / 4 test) is still far too
small to draw statistically meaningful conclusions about multimodal RAMTA's real-world
accuracy. This remains a **feasibility and preliminary-signal test**, not the full-scale
N=202 empirical evaluation described as future work in Chapter 7/9 of the thesis. Results
are reported with this caveat and are not presented as a validated performance claim.

**How to use:**
1. Runtime -> Change runtime type -> T4 GPU
2. Upload `multimodal_pilot_data_v3.zip` when prompted (Cell 2)
3. Run all cells in order.""")

code("""!pip -q install transformers faiss-cpu shap scikit-learn
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE - enable GPU runtime!')""")

code("""# Cell 2 - upload and unzip the multimodal pilot data (v3: 15 events)
from google.colab import files
up = files.upload()   # choose multimodal_pilot_data_v3.zip
import zipfile
with zipfile.ZipFile('multimodal_pilot_data_v3.zip', 'r') as z:
    z.extractall('.')
import os
print('Audio files:', sorted(os.listdir('audio')))
print('Frame dirs:', sorted(os.listdir('frames')))""")

code("""# Cell 3 - the 15 real events (text + real EUR/USD 3-day direction label,
# taken directly from experiments/data/political_events.csv used in the Chapter 7 Text+RAF pilot)
# label: 1 = EUR/USD UP over the following 3 trading days, 0 = DOWN
events = [
    {'id': 'fed_20151216', 'date': '2015-12-16', 'label': 0,
     'text': "The Federal Reserve raises interest rates for the first time in nearly a decade, "
             "signalling confidence in the US recovery while stressing a gradual path."},
    {'id': 'fed_20161214', 'date': '2016-12-14', 'label': 0,
     'text': "The Federal Reserve raises interest rates for the second time since the crisis, "
             "citing labour market strength and rising confidence in the economic outlook."},
    {'id': 'fed_20170315', 'date': '2017-03-15', 'label': 1,
     'text': "The Federal Reserve raises interest rates a third time as the recovery firms, "
             "with officials signalling a gradual pace of further increases."},
    {'id': 'fed_20171213', 'date': '2017-12-13', 'label': 1,
     'text': "The Federal Reserve raises rates in Chair Yellen's final meeting, while forecasting "
             "three more hikes in 2018 as tax cuts boost the outlook."},
    {'id': 'fed_20180321', 'date': '2018-03-21', 'label': 1,
     'text': "In his first meeting as Chair, Jerome Powell's Federal Reserve raises rates and "
             "signals a slightly steeper hiking path for the year ahead."},
    {'id': 'fed_20180926', 'date': '2018-09-26', 'label': 0,
     'text': "The Federal Reserve raises rates for the third time in 2018 and removes language "
             "describing policy as accommodative."},
    {'id': 'fed_20181219', 'date': '2018-12-19', 'label': 1,
     'text': "The Fed raises rates a fourth time in 2018 and signals further tightening, "
             "disappointing markets hoping for a dovish pivot."},
    {'id': 'fed_20190731', 'date': '2019-07-31', 'label': 0,
     'text': "The Fed cuts interest rates for the first time since 2008, describing the move as "
             "a mid-cycle adjustment rather than the start of an easing cycle."},
    {'id': 'fed_20190918', 'date': '2019-09-18', 'label': 0,
     'text': "The Federal Reserve cuts rates for a second consecutive meeting amid trade-war "
             "uncertainty, while Chair Powell resists commitment to further cuts."},
    {'id': 'fed_20211103', 'date': '2021-11-03', 'label': 1,
     'text': "The Federal Reserve announces it will begin tapering its pandemic-era bond "
             "purchases, a first step toward policy normalisation."},
    {'id': 'fed_20220316', 'date': '2022-03-16', 'label': 1,
     'text': "The Fed raises rates for the first time since 2018, beginning what it signals "
             "will be a sustained tightening cycle against inflation."},
    {'id': 'fed_20220615', 'date': '2022-06-15', 'label': 1,
     'text': "The Fed hikes rates by 75 basis points, the largest single increase since 1994, "
             "and signals more aggressive action against inflation."},
    {'id': 'fed_20230726', 'date': '2023-07-26', 'label': 1,
     'text': "The Fed raises rates to a 22-year high; Chair Powell keeps options open, saying "
             "future decisions depend on incoming data."},
    {'id': 'fed_20240918', 'date': '2024-09-18', 'label': 1,
     'text': "The Fed begins its easing cycle with an outsized 50 basis point rate cut, "
             "declaring greater confidence that inflation is moving to target."},
    {'id': 'fed_20241218', 'date': '2024-12-18', 'label': 0,
     'text': "The Federal Reserve cuts rates by 25 basis points but signals a slower pace "
             "of cuts in 2025, adopting a more hawkish tone."},
]
events = sorted(events, key=lambda e: e['date'])  # chronological order for a time-respecting split
print(f'{len(events)} real events loaded, chronologically ordered')
print('Label distribution:', sum(e[\"label\"] for e in events), 'UP /', len(events)-sum(e[\"label\"] for e in events), 'DOWN')""")

code("""# Cell 4 - Text encoder (FinBERT): FULL TOKEN SEQUENCE (not just CLS)
from transformers import AutoTokenizer, AutoModel
import torch, numpy as np

tok_text = AutoTokenizer.from_pretrained('ProsusAI/finbert')
enc_text = AutoModel.from_pretrained('ProsusAI/finbert').cuda().eval()

text_seqs = []   # list of (L_i, 768) arrays, one per event
with torch.no_grad():
    for ev in events:
        batch = tok_text(ev['text'], truncation=True, max_length=64, return_tensors='pt').to('cuda')
        out = enc_text(**batch)
        seq = out.last_hidden_state[0].cpu().numpy()   # (L, 768) - FULL sequence, not CLS only
        text_seqs.append(seq)
        print(f"  {ev['id']}: text sequence length L={seq.shape[0]}")
print('Text: kept full token sequences (no pooling before fusion)')""")

code("""# Cell 5 - Audio encoder (Wav2Vec2): FULL FRAME SEQUENCE (subsampled for tractability)
!pip -q install soundfile librosa
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import librosa

proc_audio = Wav2Vec2Processor.from_pretrained('facebook/wav2vec2-base-960h')
enc_audio = Wav2Vec2Model.from_pretrained('facebook/wav2vec2-base-960h').cuda().eval()

AUDIO_SUBSAMPLE = 25  # keep every 25th ~20ms frame (~2s stride) to keep sequence length manageable

audio_seqs = []  # list of (T_i, 768) arrays, one per event
with torch.no_grad():
    for ev in events:
        wav, sr = librosa.load(f"audio/{ev['id']}.m4a", sr=16000, duration=60)  # first 60s
        inputs = proc_audio(wav, sampling_rate=16000, return_tensors='pt').to('cuda')
        out = enc_audio(**inputs)
        full_seq = out.last_hidden_state[0]              # (T, 768), T ~ 3000 for 60s audio
        seq = full_seq[::AUDIO_SUBSAMPLE].cpu().numpy()  # subsample -> (T/25, 768)
        audio_seqs.append(seq)
        print(f"  {ev['id']}: audio sequence length T={seq.shape[0]} (subsampled from {full_seq.shape[0]})")
print('Audio: kept subsampled frame sequences (no mean-pooling before fusion)')""")

code("""# Cell 6 - Visual encoder (ViT): FULL KEY-FRAME SEQUENCE (18 frames per event)
from transformers import ViTImageProcessor, ViTModel
from PIL import Image
import glob

proc_vis = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
enc_vis = ViTModel.from_pretrained('google/vit-base-patch16-224').cuda().eval()

visual_seqs = []  # list of (18, 768) arrays, one per event
with torch.no_grad():
    for ev in events:
        frame_paths = sorted(glob.glob(f"frames/{ev['id']}/*.jpg"))
        frame_vecs = []
        for fp in frame_paths:
            img = Image.open(fp).convert('RGB')
            inputs = proc_vis(images=img, return_tensors='pt').to('cuda')
            out = enc_vis(**inputs)
            frame_vecs.append(out.last_hidden_state[0, 0, :].cpu().numpy())  # CLS token per frame
        seq = np.stack(frame_vecs)  # (18, 768) - full frame sequence, not mean-pooled
        visual_seqs.append(seq)
        print(f"  {ev['id']}: visual sequence length = {seq.shape[0]} key-frames")
print('Visual: kept per-frame sequences (no mean-pooling before fusion)')""")

code("""# Cell 7 - TRUE sequence-level cross-modal attention fusion (Section 5.5 formulation)
# CrossAttn(A, B) = Softmax( Q_A K_B^T / sqrt(d_k) ) V_B
# Q_A: (L_A, 768) query sequence from modality A: K_B, V_B: (L_B, 768) from modality B.
# Softmax is over the L_B context positions -> non-degenerate whenever L_B > 1.
import torch.nn.functional as F

d_k = 768

def cross_attn_sequence(query_seq, kv_seq):
    \"\"\"query_seq: (L_A, 768) numpy; kv_seq: (L_B, 768) numpy. Returns pooled (768,) output
    and the (L_A, L_B) attention weight matrix for inspection.\"\"\"
    Q = torch.tensor(query_seq, dtype=torch.float32).cuda()   # (L_A, 768)
    K = torch.tensor(kv_seq, dtype=torch.float32).cuda()      # (L_B, 768)
    V = K.clone()
    scores = (Q @ K.T) / (d_k ** 0.5)                         # (L_A, L_B)
    weights = F.softmax(scores, dim=-1)                       # softmax over L_B (context) positions
    attended = weights @ V                                    # (L_A, 768)
    pooled = attended.mean(dim=0)                              # mean-pool over L_A (query) -> (768,)
    return pooled.cpu().numpy(), weights.cpu().numpy()

fused_list = []
example_weights = None
entropies = []
for i, ev in enumerate(events):
    T_seq, A_seq, V_seq = text_seqs[i], audio_seqs[i], visual_seqs[i]

    ta, w_ta = cross_attn_sequence(T_seq, A_seq)
    at, w_at = cross_attn_sequence(A_seq, T_seq)
    tv, w_tv = cross_attn_sequence(T_seq, V_seq)
    vt, w_vt = cross_attn_sequence(V_seq, T_seq)
    av, w_av = cross_attn_sequence(A_seq, V_seq)
    va, w_va = cross_attn_sequence(V_seq, A_seq)

    fused_raw_i = np.concatenate([ta, at, tv, vt, av, va])   # (4608,)
    fused_list.append(fused_raw_i)

    if i == 0:
        example_weights = w_tv   # (L_T, L_V) attention matrix, Text attending to Visual

    # Non-degeneracy check: with L_B > 1 keys, entropy of the softmax distribution is > 0
    ent = -(w_tv * np.log(w_tv + 1e-12)).sum(axis=-1).mean()
    entropies.append(ent)
    max_ent = np.log(w_tv.shape[-1])
    print(f"  {ev['id']}: T->V attention over L_B={w_tv.shape[-1]} visual frames, "
          f"mean entropy={ent:.3f} (max possible={max_ent:.3f}, 0 would indicate a degenerate/one-hot distribution)")

fused_raw = np.stack(fused_list)  # (15, 4608)
print()
print('Fused (pre-projection) shape:', fused_raw.shape)
print(f'Mean T->V attention entropy across all {len(events)} events: {np.mean(entropies):.3f} (all > 0 => non-degenerate)')
print()
print('Example Text->Visual attention weight matrix (first event), rows=text tokens, cols=video frames:')
print(np.round(example_weights[:5], 3))  # first 5 text-token rows for inspection

# Linear projection to 1024-dim (Section 5.5); random-init, not trained (no gradient signal defined yet)
torch.manual_seed(42)
proj = torch.nn.Linear(fused_raw.shape[1], 1024).cuda()
with torch.no_grad():
    fused = proj(torch.tensor(fused_raw, dtype=torch.float32).cuda()).cpu().numpy()
print()
print('F_fused shape:', fused.shape)""")

code("""# Cell 8 - minimal real train/test evaluation (N=15: 11 train / 4 test, chronological split)
# This is a preliminary quantitative signal, NOT a statistically powered evaluation.
# Chronological split avoids look-ahead: test events all occur after all train events.
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

labels = np.array([e['label'] for e in events])
n_test = 4
X_train, X_test = fused[:-n_test], fused[-n_test:]
y_train, y_test = labels[:-n_test], labels[-n_test:]

scaler = StandardScaler().fit(X_train)
X_train_s, X_test_s = scaler.transform(X_train), scaler.transform(X_test)

clf = LogisticRegression(max_iter=2000, C=0.1).fit(X_train_s, y_train)
pred = clf.predict(X_test_s)
acc = (pred == y_test).mean()

print(f'Train events: {[e[\"date\"] for e in events[:-n_test]]}')
print(f'Test events:  {[e[\"date\"] for e in events[-n_test:]]}')
print(f'Test true labels:      {y_test.tolist()}')
print(f'Test predicted labels: {pred.tolist()}')
print(f'Test-set directional accuracy: {acc:.3f} ({int(acc*n_test)}/{n_test})')
print()
print('CAVEAT: with only 4 test events, this accuracy figure has enormous sampling')
print('uncertainty (a 95% Wilson CI for k/4 correct spans roughly 30-100 percentage points')
print('depending on k) and must NOT be read as a validated performance claim. It is reported')
print('only as a preliminary existence-of-signal check alongside the architecture-feasibility')
print('result above. The statistically powered evaluation remains the N=202 Text+RAF pilot')
print('(Chapter 7, Table 7.2) and, as future work, a full-scale multimodal replication.')

# Wilson 95% CI for the observed test accuracy
from math import sqrt
def wilson_ci(k, n, z=1.96):
    if n == 0: return (float('nan'), float('nan'))
    p = k / n
    denom = 1 + z**2/n
    centre = p + z**2/(2*n)
    adj = z * sqrt(p*(1-p)/n + z**2/(4*n**2))
    return ((centre - adj)/denom, (centre + adj)/denom)

lo, hi = wilson_ci(int(acc*n_test), n_test)
print(f'95% Wilson CI for test accuracy: [{lo:.3f}, {hi:.3f}]')""")

code("""# Cell 9 - FAISS retrieval demonstration (real retrieval over the 15-event set)
import faiss

Fn = fused / np.linalg.norm(fused, axis=1, keepdims=True)
index = faiss.IndexFlatIP(Fn.shape[1])
index.add(Fn.astype('float32'))

print('Leave-one-out nearest-neighbour retrieval among the 15 multimodal events:')
for i, ev in enumerate(events):
    sims, idxs = index.search(Fn[i:i+1].astype('float32'), 2)  # top-2 (self + nearest other)
    j = idxs[0][1] if idxs[0][0] == i else idxs[0][0]
    sim = sims[0][1] if idxs[0][0] == i else sims[0][0]
    print(f"  {ev['date']} ({ev['text'][:40]}...)")
    print(f"    -> nearest: {events[j]['date']} (sim={sim:.3f})")""")

code("""# Cell 10 - SHAP explainability on the fused multimodal representation
import shap

explainer = shap.LinearExplainer(clf, X_train_s)
shap_vals = explainer.shap_values(X_test_s)
print('SHAP values shape:', shap_vals.shape, '(N_test events x 1024 fused dims)')
print('Aggregate mean |SHAP| across all fused dims:', np.abs(shap_vals).mean())""")

code("""# Cell 11 - save results and download
import pandas as pd
summary = pd.DataFrame({
    'event_date': [e['date'] for e in events],
    'label_UP': [e['label'] for e in events],
    'text_seq_len': [s.shape[0] for s in text_seqs],
    'audio_seq_len': [s.shape[0] for s in audio_seqs],
    'visual_seq_len': [s.shape[0] for s in visual_seqs],
    'fused_dim': [fused.shape[1]]*len(events),
    'tv_attn_entropy': entropies,
})
summary.to_csv('multimodal_pilot_summary_v3.csv', index=False)
print(summary.to_string(index=False))

eval_summary = pd.DataFrame({
    'n_train': [len(y_train)], 'n_test': [len(y_test)],
    'test_accuracy': [acc], 'wilson_ci_low': [lo], 'wilson_ci_high': [hi],
})
eval_summary.to_csv('multimodal_pilot_eval_v3.csv', index=False)
print(eval_summary.to_string(index=False))

np.savez('multimodal_pilot_embeddings_v3.npz', fused=fused, labels=labels)

from google.colab import files
files.download('multimodal_pilot_summary_v3.csv')
files.download('multimodal_pilot_eval_v3.csv')
files.download('multimodal_pilot_embeddings_v3.npz')""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

out = r'C:\Users\melik\OneDrive\Desktop\newthesis\experiments\RAMTA_multimodal_pilot_v3_colab.ipynb'
with io.open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('notebook written:', out)
