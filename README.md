# RAMTA Experiments — Code Repository

Thesis: *Analysis of the Impact of Political Multimodal Discourse on Financial Markets
Using Retrieval-Augmented and Explainable Transformer Architectures*
Melike Nazlı Gürbüz (s34853), PJAIT

This repository contains all code used to produce the empirical results reported in
Chapter 7 of the thesis: the Text + RAF pilot (Section 7.6, 79→202 event dataset
progression) and the multimodal feasibility test (Section 7.5.1, N=5 real events).

## Structure

```
experiments/
├── 01_download_financial_data.py       (4 KB)   Downloads EUR/USD, GBP/USD, USD/JPY, VIX (2015-2025) via yfinance
├── 02_build_event_dataset.py           (48 KB)  Curated 202-event political dataset + real market-based labels
├── 03_econometric_baselines.py         (8 KB)   ARIMA, GARCH(1,1) benchmarks, ADF/Phillips-Perron stationarity tests
├── RAMTA_experiments_colab.ipynb       (12 KB)  Text + RAF pilot: FinBERT embeddings, FAISS retrieval, SHAP (Section 7.6)
├── RAMTA_multimodal_pilot_colab.ipynb  (16 KB)  Full multimodal feasibility test: FinBERT+Wav2Vec2+ViT+cross-attention+FAISS+SHAP (Section 7.5.1)
├── build_notebook.py                   (12 KB)  Generates RAMTA_experiments_colab.ipynb
├── build_multimodal_notebook.py        (12 KB)  Generates RAMTA_multimodal_pilot_colab.ipynb
├── data/                                (889 KB total)
│   ├── financial_data.csv              (796 KB, 2,712 rows)  Daily OHLC + log returns + rolling volatility, 2015-2025
│   ├── political_events.csv            (88 KB, 202 rows)     202 curated political events with real market-reaction labels
│   └── results_econometric.csv         (1 KB)                ARIMA/GARCH benchmark output
└── media/
    └── frames/                         (5.5 MB, 90 JPEGs)    Key-frames (1 fps/10s, 18 per event × 5 events) extracted
                                                                 from the N=5 feasibility test videos

Total tracked in this repository: ~6.5 MB
Excluded (see .gitignore): raw audio (9.8 MB) and video (109 MB) source files,
~119 MB total — see "Multimodal feasibility test" section below for how to
re-obtain them.
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

The five real political events used are Federal Reserve press conferences
(16 Dec 2015, 19 Dec 2018, 16 Mar 2022, 18 Sep 2024, 18 Dec 2024), with audio
and video sourced directly from the Federal Reserve's own public webcast
archive at:

```
https://www.federalreserve.gov/monetarypolicy/fomcpresconfYYYYMMDD.htm
```

(replace YYYYMMDD with each date above). Raw audio/video files (~133 MB total)
are not included in this repository due to size; `media/frames/` contains the
extracted key-frames (1 frame per 10 seconds, first 3 minutes of each press
conference) actually used to produce the results in Section 7.5.1. To
re-download the source media and reproduce frame/audio extraction from
scratch, `yt-dlp` (with `ffmpeg`) can be pointed at the URLs above; the exact
commands used are documented in the thesis text (Section 7.5.1) and available
on request.

Upload `RAMTA_multimodal_pilot_colab.ipynb` to Google Colab (T4 GPU runtime)
together with the audio files and `media/frames/` (zipped) to reproduce the
end-to-end multimodal pipeline execution (FinBERT + Wav2Vec2 + ViT +
cross-modal fusion + FAISS + SHAP) described in Section 7.5.1.

## Notes on methodology

- All models use **frozen** pre-trained encoders (FinBERT, Wav2Vec2-base,
  ViT-Base/16) with lightweight classification heads, not full fine-tuning —
  see thesis Section 4.2 for the rationale (dataset size relative to model
  parameter count).
- The cross-modal attention fusion used in the N=5 feasibility test operates
  on **pooled** modality vectors rather than full token/frame sequences; this
  is a known simplification, documented and discussed in thesis Section 5.5
  and Section 7.5.1. The corrected sequence-level formulation is specified in
  Section 5.5 as the target design for the full-scale evaluation.
- All directional-accuracy results are reported with 95% Wilson confidence
  intervals (see thesis Section 7.6, Table 7.2) given the small test-set
  sizes involved at the current stage of the thesis.
