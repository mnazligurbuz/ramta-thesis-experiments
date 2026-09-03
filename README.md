# RAMTA Experiments — Code Repository

Thesis: *Analysis of the Impact of Political Multimodal Discourse on Financial Markets
Using Retrieval-Augmented and Explainable Transformer Architectures*
Melike Nazlı Gürbüz (s34853), PJAIT

This repository contains all code used to produce the empirical results reported in
Chapter 7 of the thesis: the Text + RAF pilot (Section 7.6, 79→202 event dataset
progression) and the multimodal feasibility test (Section 7.5.1, N=15 real events,
extended from an initial N=5 pilot).

## Structure

```
experiments/
├── 01_download_financial_data.py       (4 KB)   Downloads EUR/USD, GBP/USD, USD/JPY, VIX (2015-2025) via yfinance
├── 02_build_event_dataset.py           (48 KB)  Curated 202-event political dataset + real market-based labels
├── 03_econometric_baselines.py         (8 KB)   ARIMA, GARCH(1,1) benchmarks, ADF/Phillips-Perron stationarity tests
├── RAMTA_experiments_colab.ipynb          (12 KB)  Text + RAF pilot: FinBERT embeddings, FAISS retrieval, SHAP (Section 7.6)
├── RAMTA_multimodal_pilot_colab.ipynb     (16 KB)  N=5 multimodal feasibility test (pooled-vector attention; superseded by v2/v3)
├── RAMTA_multimodal_pilot_v2_colab.ipynb  (16 KB)  N=5, true sequence-level cross-attention (Section 5.5 formulation)
├── RAMTA_multimodal_pilot_v3_colab.ipynb  (18 KB)  N=15, sequence-level cross-attention + minimal real train/test evaluation (Section 7.5.1, current)
├── build_notebook.py                      (12 KB)  Generates RAMTA_experiments_colab.ipynb
├── build_multimodal_notebook.py           (12 KB)  Generates RAMTA_multimodal_pilot_v2_colab.ipynb
├── build_multimodal_notebook_v3.py        (14 KB)  Generates RAMTA_multimodal_pilot_v3_colab.ipynb
├── 04_threshold_sensitivity.py            (2 KB)   0.1% neutral-band threshold sensitivity analysis (Section 3.2.1/4.4)
├── data/                                   (889 KB total)
│   ├── financial_data.csv                 (796 KB, 2,712 rows)  Daily OHLC + log returns + rolling volatility, 2015-2025
│   ├── political_events.csv               (88 KB, 202 rows)     202 curated political events with real market-reaction labels
│   ├── results_econometric.csv            (1 KB)                ARIMA/GARCH benchmark output
│   └── results_threshold_sensitivity.csv  (1 KB)                Label-threshold sensitivity results
└── media/
    └── frames/                            (16.5 MB, 270 JPEGs)  Key-frames (1 fps/10s, 18 per event × 15 events)

Total tracked in this repository: ~17.5 MB
Excluded (see .gitignore): raw audio (~150 MB) and video (~360 MB) source files
for all 15 events, and the packaged multimodal_pilot_data_v3.zip (~118 MB) used
to upload data to Colab — see "Multimodal feasibility test" section below for
how to re-obtain them.
```

## Reproducing the results

### Text + RAF pilot (Section 7.6, Table 7.2)

```bash
pip install yfinance pandas numpy statsmodels arch
python 01_download_financial_data.py
python 02_build_event_dataset.py
python 03_econometric_baselines.py
```

Then upload `RAMTA_experiments_colab.ipynb` to Google Colab (T4 GPU runtime),
and upload `data/political_events.csv` when prompted in Cell 2. This reproduces
the FinBERT text-only and FAISS-augmented (RAF) directional accuracy results in
Table 7.2, and the SHAP attribution plot in Figure 7.2.

### Multimodal feasibility test (Section 7.5.1)

The fifteen real political events used are Federal Reserve press conferences
(16 Dec 2015, 14 Dec 2016, 15 Mar 2017, 13 Dec 2017, 21 Mar 2018, 26 Sep 2018,
19 Dec 2018, 31 Jul 2019, 18 Sep 2019, 3 Nov 2021, 16 Mar 2022, 15 Jun 2022,
26 Jul 2023, 18 Sep 2024, 18 Dec 2024), with audio and video sourced directly
from the Federal Reserve's own public webcast archive at:

```
https://www.federalreserve.gov/monetarypolicy/fomcpresconfYYYYMMDD.htm
```

(replace YYYYMMDD with each date above). Raw audio/video files (~510 MB total
across all 15 events) are not included in this repository due to size;
`media/frames/` contains the extracted key-frames (1 frame per 10 seconds,
first 3 minutes of each press conference) actually used to produce the
results in Section 7.5.1. To re-download the source media and reproduce
frame/audio extraction from scratch, `yt-dlp` (with `ffmpeg`) can be pointed
at the URLs above; the exact commands used are documented in the thesis text
(Section 7.5.1) and available on request.

Upload `RAMTA_multimodal_pilot_v3_colab.ipynb` to Google Colab (T4 GPU
runtime) together with `multimodal_pilot_data_v3.zip` (audio + `media/frames/`
for all 15 events) to reproduce the end-to-end multimodal pipeline execution
(FinBERT + Wav2Vec2 + ViT + sequence-level cross-modal fusion + a minimal
11-train/4-test directional evaluation + FAISS + SHAP) described in
Section 7.5.1. The earlier `RAMTA_multimodal_pilot_colab.ipynb` (N=5, pooled
attention) and `RAMTA_multimodal_pilot_v2_colab.ipynb` (N=5, sequence-level
attention) are retained in this repository for provenance but are superseded
by v3.

Actual output from running `RAMTA_multimodal_pilot_v3_colab.ipynb` end-to-end is
saved in `data/multimodal_pilot_summary_v3.csv` (per-event sequence lengths and
T->V attention entropy, confirming non-degeneracy for all 15 events),
`data/multimodal_pilot_eval_v3.csv` (test accuracy = 0.75, 3/4 correct, 95%
Wilson CI [0.301, 0.954]), and `data/multimodal_pilot_embeddings_v3.npz` (the
1024-dim fused representations and labels for all 15 events).

## Notes on methodology

- All models use **frozen** pre-trained encoders (FinBERT, Wav2Vec2-base,
  ViT-Base/16) with lightweight classification heads, not full fine-tuning —
  see thesis Section 4.2 for the rationale (dataset size relative to model
  parameter count).
- The cross-modal attention fusion used in v2/v3 operates on **true
  sequence-level** token/frame/audio-segment representations (Section 5.5
  formulation), not pooled single vectors; non-degeneracy is verified via
  attention-entropy diagnostics printed in the notebook (Section 7.5.1). The
  earlier N=5 pooled-vector version (`RAMTA_multimodal_pilot_colab.ipynb`) is
  kept only for provenance and is explicitly labelled as superseded.
- v3 reports a minimal real directional-accuracy result on a chronological
  11-train/4-test split of the N=15 event set. This is a **preliminary
  signal check, not a statistically powered evaluation** — with only 4 test
  events the 95% Wilson confidence interval is very wide, as stated directly
  in the notebook output and in thesis Section 7.5.1. The statistically
  powered evaluation remains the N=202 Text+RAF pilot (Section 7.6, Table 7.2).
- All directional-accuracy results are reported with 95% Wilson confidence
  intervals given the small test-set sizes involved at the current stage of
  the thesis.
