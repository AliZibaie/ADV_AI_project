# Air Pollution Forecasting with Deep Learning Models

## Project Overview

This project focuses on forecasting air pollution (PM2.5 concentration) using deep learning models. Two recurrent neural network architectures — **LSTM** and **GRU** — are trained and compared against a naive baseline on hourly time-series data from Beijing.

---

## Dataset

| Attribute | Details |
|-----------|---------|
| **Name** | Beijing PM2.5 Data (PRSA Dataset) |
| **File** | `PRSA_data_2010.1.1-2014.12.31.csv` |
| **Period** | January 2010 – December 2014 |
| **Rows** | ~43,824 (hourly records) |
| **Source** | UCI Machine Learning Repository |

### Features Used

| Feature | Description |
|---------|-------------|
| `pm2.5` | PM2.5 concentration — **prediction target** |
| `DEWP` | Dew point temperature |
| `TEMP` | Air temperature |
| `PRES` | Atmospheric pressure |
| `Iws` | Cumulative wind speed |
| `Is` | Cumulative hours of snow |
| `Ir` | Cumulative hours of rain |

---

## Folder Structure

```
codes/
├── index.ipynb                              # Main notebook (all phases)
├── README.md                                # Project guide
├── .gitignore
│
├── dataset/
│   ├── PRSA_data_2010.1.1-2014.12.31.csv   # Raw dataset
│   └── processed_sequences.npz             # Preprocessed sequences (auto-generated)
│
├── outputs/
│   ├── lstm_model.h5                        # Saved LSTM model weights
│   └── model_gru.h5                         # Saved GRU model weights
│
└── results/
    ├── eda_analysis.png                     # Exploratory data analysis plots
    ├── correlation_matrix.png               # Feature correlation heatmap
    ├── preprocessing_split_missing.png      # Missing values & data split analysis
    ├── preprocessing_transform_scaling.png  # Normalization visualization
    ├── model_evaluation_comparison.png      # Model comparison charts
    └── model_comparison.csv                 # Numeric evaluation metrics
```

---

## Requirements

### System
- Python **3.8+**
- Jupyter Notebook or JupyterLab(optional)
- At least **4 GB RAM**

### Libraries

Install all dependencies with a single command:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow scipy
```

Or pin specific versions for reproducibility:

```bash
pip install pandas==2.0.3 numpy==1.24.3 matplotlib==3.7.2 seaborn==0.12.2 \
            scikit-learn==1.3.0 tensorflow==2.13.0 scipy==1.11.2
```

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/AliZibaie/ADV_AI_project
cd ADV_AI_project
```

**2. Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn tensorflow scipy
```

**3. Run the project**

You have two options:

### Option A: Run as Python script (recommended)
```bash
python index.py
```

### Option B: Run in Jupyter Notebook
```bash
jupyter notebook
```
Then open `index.ipynb` and run cells top to bottom.

> **Note:** If `processed_sequences.npz` already exists in `dataset/`, Phase 2 will skip re-processing and load it directly, saving time.

---

## Notebook Phases

The notebook is organized into five sequential phases:

### Phase 1 — Exploratory Data Analysis (EDA)
- Load and inspect the raw dataset
- Statistical summary of all features
- Correlation matrix and distribution plots
- Outlier detection

### Phase 2 — Data Preprocessing
- Handle missing values (imputation/removal)
- Feature standardization with `StandardScaler`
- Build sliding-window time-series sequences (window = 24 hours)
- Split data into Train / Validation / Test sets
- Save processed sequences to `dataset/processed_sequences.npz`

### Phase 3 — Model Architecture & Training
- Build **LSTM** model (`Sequential` with LSTM + Dense layers)
- Build **GRU** model (`Sequential` with GRU + Dense layers)
- Train both models with `EarlyStopping` callback

**Key hyperparameters:**

| Parameter | Value |
|-----------|-------|
| Sequence length | 24 (hours) |
| Batch size | 32 |
| Max epochs | 100 |
| Optimizer | Adam |
| Loss function | MSE |

### Phase 4 — Evaluation & Comparison
- Evaluate on the held-out test set
- Metrics: MAE, RMSE, MAPE, R²
- Compare LSTM and GRU against the Baseline (Naive/last-value) model
- Generate comparison plots saved to `results/`

### Phase 5 — Export Model Weights
- Save trained LSTM model to `outputs/lstm_model.h5`
- Save trained GRU model to `outputs/model_gru.h5`

---

## Results

| Model | MAE | RMSE | MAPE (%) | R² |
|-------|-----|------|----------|----|
| Baseline (Naive) | 0.1736 | 0.2760 | 94.27 | 0.9322 |
| LSTM | **0.1655** | **0.2545** | 84.79 | **0.9424** |
| GRU | 0.1672 | 0.2555 | **84.75** | 0.9419 |

**Key takeaways:**
- Both LSTM and GRU outperform the naive baseline on all metrics.
- LSTM achieves the lowest MAE and RMSE; GRU achieves a marginally lower MAPE.
- The performance gap between LSTM and GRU is minimal (~1%).

---

## Re-training from Scratch

To force a full re-run including preprocessing:

```bash
rm dataset/processed_sequences.npz
rm outputs/*.h5
```

Then re-run all cells in `index.ipynb`.
