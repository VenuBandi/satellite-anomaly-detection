# Satellite Telemetry Anomaly Detection

LSTM Autoencoder anomaly detection on NASA SMAP and MSL spacecraft telemetry, with MLflow experiment tracking and threshold sweep against ground-truth labels.

## Dataset

NASA SMAP (Soil Moisture Active Passive) and MSL (Mars Science Laboratory / Curiosity Rover) telemetry.  
55 SMAP channels + 27 MSL channels · 105 labeled anomaly sequences · Released by NASA JPL.  
Source: [khundman/telemanom](https://github.com/khundman/telemanom) · Reference: [arXiv:1802.04431](https://arxiv.org/abs/1802.04431)

## Approach

```
NASA telemetry (.npy per channel)
    └── LSTM Autoencoder trained on nominal (non-anomalous) sequences
        └── Reconstruction error on test window
            └── Dynamic threshold (percentile sweep + MAD)
                └── Anomaly spans vs. labeled ground truth
                    └── Precision / Recall / F1 per channel
```

## Quickstart

```bash
pip install -r requirements.txt
cd src
python data_loader.py          # downloads 5 representative channels
python train_detector.py       # trains LSTM AE, logs to MLflow
python evaluate.py             # threshold sweep, plots anomaly spans
```

## Results

| Channel | F1 | Precision | Recall |
|---|---|---|---|
| A-1 | — | — | — |
| D-14 | — | — | — |

*Run `python evaluate.py` to populate results for your environment.*

## Stack

Python · PyTorch · XGBoost · SHAP · MLflow · NumPy · pandas · matplotlib
