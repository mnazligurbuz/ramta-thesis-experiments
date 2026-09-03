# -*- coding: utf-8 -*-
"""Build the Colab notebook for the small real multimodal (Text+Audio+Visual+RAF) feasibility pilot.
v2: uses TRUE sequence-level cross-attention (token/frame/audio-segment sequences),
not pooled-vector cross-attention, per Section 5.5 of the thesis and the supervisor's
comment on the cross-attention formulation."""
import json, io

cells = []
def md(src): cells.append({"cell_type":"markdown","metadata":{},"source":src.splitlines(keepends=True)})
def code(src): cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":src.splitlines(keepends=True)})

md("""# RAMTA Multimodal Feasibility Pilot v2 (Text + Audio + Visual + RAF)
**Thesis:** Analysis of the Impact of Political Multimodal Discourse on Financial Markets — Melike Nazlı Gürbüz (s34853)

This notebook implements and runs the **full multimodal RAMTA architecture end-to-end**
on N=5 real political events (FOMC press conferences, 2015-2024), each with real
text, audio, and video sourced from federalreserve.gov.

**v2 change:** this version implements **true sequence-level cross-attention**
(operating over token / audio-frame / video-frame sequences, as specified in
Section 5.5 of the thesis) rather than the pooled-single-vector cross-attention
used in the v1 feasibility test. This directly addresses the mathematical
degeneracy concern raised for v1 (softmax over a single key always equals 1).

Because N=5 is too small for a statistically powered accuracy evaluation, this
notebook remains a **feasibility/implementation demonstration**: it shows that
every component of the proposed architecture (FinBERT text encoder, Wav2Vec2
audio encoder, ViT visual encoder, sequence-level cross-modal fusion, FAISS
retrieval, SHAP explainability) runs correctly end-to-end on real data, and
verifies non-degenerate attention weights. Large-scale statistical evaluation
remains future work, consistent with Chapter 7 of the thesis.

**How to use:**
1. Runtime -> Change runtime type -> T4 GPU
2. Upload `multimodal_pilot_data.zip` when prompted (Cell 2)
3. Run all cells in order.""")

code("""!pip -q install transformers faiss-cpu shap scikit-learn
import torch
print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NOT AVAILABLE - enable GPU runtime!')""")

code("""# Cell 2 - upload and unzip the multimodal pilot data
from google.colab import files
up = files.upload()   # choose multimodal_pilot_data.zip
import zipfile
with zipfile.ZipFile('multimodal_pilot_data.zip', 'r') as z:
    z.extractall('.')
import os
print('Audio files:', os.listdir('audio'))
print('Frame dirs:', os.listdir('frames'))""")

code("""# Cell 3 - the 5 real events (text matches thesis political_events.csv)
events = [
    {'id': 'fed_20151216', 'date': '2015-12-16',
     'text': "The Federal Reserve raises interest rates for the first time in nearly a decade, "
             "signalling confidence in the US recovery while stressing a gradual path."},
    {'id': 'fed_20181219', 'date': '2018-12-19',
     'text': "The Fed raises rates a fourth time in 2018 and signals further tightening, "
             "disappointing markets hoping for a dovish pivot."},
    {'id': 'fed_20220316', 'date': '2022-03-16',
     'text': "The Fed raises rates for the first time since 2018, beginning what it signals "
             "will be a sustained tightening cycle against inflation."},
    {'id': 'fed_20240918', 'date': '2024-09-18',
     'text': "The Fed begins its easing cycle with an outsized 50 basis point rate cut, "
             "declaring greater confidence that inflation is moving to target."},
    {'id': 'fed_20241218', 'date': '2024-12-18',
     'text': "The Federal Reserve cuts rates by 25 basis points but signals a slower pace "
             "of cuts in 2025, adopting a more hawkish tone."},
]
print(f'{len(events)} real events loaded')""")

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
    max_ent = np.log(w_tv.shape[-1])
    print(f"  {ev['id']}: T->V attention over L_B={w_tv.shape[-1]} visual frames, "
          f"mean entropy={ent:.3f} (max possible={max_ent:.3f}, 0 would indicate a degenerate/one-hot distribution)")

fused_raw = np.stack(fused_list)  # (5, 4608)
print()
print('Fused (pre-projection) shape:', fused_raw.shape)
print()
print('Example Text->Visual attention weight matrix (first event), rows=text tokens, cols=video frames:')
print(np.round(example_weights[:5], 3))  # first 5 text-token rows for inspection
print()
print('Because each modality is now represented as a genuine multi-element sequence')
print('(not a single pooled vector), the softmax above is computed over multiple keys')
print('and produces a non-trivial (non-degenerate) attention distribution, unlike the')
print('pooled-vector version used in the v1 feasibility test.')

# Linear projection to 1024-dim (Section 5.5), random-init since no training data at N=5
torch.manual_seed(42)
proj = torch.nn.Linear(fused_raw.shape[1], 1024).cuda()
with torch.no_grad():
    fused = proj(torch.tensor(fused_raw, dtype=torch.float32).cuda()).cpu().numpy()
print()
print('F_fused shape:', fused.shape)
print('NOTE: with N=5 events there is no independent training set to learn the fusion')
print('projection weights; this cell demonstrates that the (now non-degenerate)')
print('sequence-level cross-attention mechanism executes correctly end-to-end on real')
print('data, not a trained/optimised model.')""")

code("""# Cell 8 - FAISS retrieval demonstration (real retrieval over the 5-event set)
import faiss

Fn = fused / np.linalg.norm(fused, axis=1, keepdims=True)
index = faiss.IndexFlatIP(Fn.shape[1])
index.add(Fn.astype('float32'))

print('Leave-one-out nearest-neighbour retrieval among the 5 multimodal events:')
for i, ev in enumerate(events):
    sims, idxs = index.search(Fn[i:i+1].astype('float32'), 2)  # top-2 (self + nearest other)
    j = idxs[0][1] if idxs[0][0] == i else idxs[0][0]
    sim = sims[0][1] if idxs[0][0] == i else sims[0][0]
    print(f"  {ev['date']} ({ev['text'][:40]}...)")
    print(f"    -> nearest: {events[j]['date']} (sim={sim:.3f})")""")

code("""# Cell 9 - SHAP explainability on the fused multimodal representation
import shap

# Use a simple linear probe (financial direction is not fit at N=5; this demonstrates
# that SHAP runs correctly on the true 1024-dim fused multimodal representation)
from sklearn.linear_model import LogisticRegression
dummy_labels = np.array([1,0,1,0,1])  # placeholder direction labels for architecture demo only
clf = LogisticRegression(max_iter=1000).fit(fused, dummy_labels)

explainer = shap.LinearExplainer(clf, fused)
shap_vals = explainer.shap_values(fused)
print('SHAP values shape:', shap_vals.shape, '(N events x 1024 fused dims)')
print('Aggregate mean |SHAP| across all fused dims:', np.abs(shap_vals).mean())""")

code("""# Cell 10 - save results and download
import pandas as pd
summary = pd.DataFrame({
    'event_date': [e['date'] for e in events],
    'text_seq_len': [s.shape[0] for s in text_seqs],
    'audio_seq_len': [s.shape[0] for s in audio_seqs],
    'visual_seq_len': [s.shape[0] for s in visual_seqs],
    'fused_dim': [fused.shape[1]]*5,
})
summary.to_csv('multimodal_pilot_summary_v2.csv', index=False)
print(summary.to_string(index=False))

np.savez('multimodal_pilot_embeddings_v2.npz', fused=fused)

from google.colab import files
files.download('multimodal_pilot_summary_v2.csv')
files.download('multimodal_pilot_embeddings_v2.npz')""")

nb = {"cells": cells,
      "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                   "kernelspec": {"display_name": "Python 3", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 0}

out = r'C:\Users\melik\OneDrive\Desktop\newthesis\experiments\RAMTA_multimodal_pilot_v2_colab.ipynb'
with io.open(out, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('notebook written:', out)
