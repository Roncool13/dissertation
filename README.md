# Multimodal Stock Movement Prediction for Indian Equities  
*A Dissertation Project – M.Tech (AI/ML)*

---

## 📌 Project Overview

This project aims to build a **multimodal deep learning system** that predicts **short-term stock price direction** for Indian equities (NSE).  
Instead of relying on a single data type, the system fuses diverse signals from:

- **Tabular OHLCV data** with technical indicators  
- **Price-action candlestick patterns**  
- **Market structure** (swing highs/lows, HH/HL/LH/LL)  
- *(Planned)* Candlestick chart images  
- *(Optional)* Textual/sentiment signals  

The end goal is a **late-fusion meta-model** that ingests predictions from each modality and outputs a final robust prediction of whether the stock will move **UP** or **NOT-UP** over a short horizon.

---

## 🎯 Objectives

### 1. Primary Prediction Task
Predict **5-day stock price direction** (binary: Up / Not-Up).  
This horizon provides a smoother, more reliable signal for pattern and structure-based models.

### 2. Baseline Task
Predict **1-day direction** as a baseline.  
This aligns with the originally submitted abstract and helps compare horizons.

### 3. Multimodal System
Train independent modality-specific models:
- A **tabular model** on OHLCV + indicators  
- A **vision model** on candlestick images  
- A **sequence/structure model** on swing structure  

Then combine their probability outputs in a **meta-classifier** for final predictions.

---

## 🧠 Approach

### Step 1 — Data Layer  
Fetch accurate daily OHLCV data from NSE using **jugaad-data**  
(works with the new NSE site, unlike deprecated nsepy).

### Step 2 — Feature Engineering  
Compute a rich set of price-action and technical features:

#### Technical Indicators
- EMA(20/50/200)  
- MACD + Signal + Histogram  
- Bollinger Bands  
- ATR(14)  
- Daily returns & 20-day volatility  

#### Candlestick Pattern Flags
- Bullish/Bearish Engulfing  
- Hammer  
- Shooting Star  
- Doji  

#### Market Structure Labels
- Swing highs and lows  
- Structural trend tags: HH, HL, LH, LL  

#### Targets
- **target_up_1d** → next-day direction  
- **target_up_5d** → next 5-day direction (primary)

### Step 3 — Fusion Model  
Use:
- `p_up_tabular`  
- `p_up_image`  
- `p_up_structure`  

as inputs to a meta-model to produce the final decision.

---

## 📂 Repository Structure
project/
│
├── pipeline.ipynb               # Data pipeline notebook (complete)
├── data/
│   └── ohlcv_ml_ready.parquet   # Generated ML-ready dataset
│
├── models/
│   ├── tabular/                 # Tabular ML models (future)
│   ├── vision/                  # CNN candlestick models (future)
│   └── structure/               # Swing/structure sequence models (future)
│
├── fusion/
│   └── meta_model.py            # Late-fusion model (future)
│
└── README.md                    # This file

---

## 🔧 Pipeline Summary (Already Implemented)

### ✔ 1. Download OHLCV  
Using:

```python
from jugaad_data.nse import stock_df

✓ Reliable
✓ NSE-compatible
✓ No SSL errors
```

### ✔ 2. Clean Data
	•	Convert dtypes
	•	Drop invalid prices
	•	Sort by symbol, date
	•	Remove duplicates

### ✔ 3. Add Technical Indicators

EMA, MACD, Bollinger Bands, ATR, returns, volatility.

### ✔ 4. Add Candlestick Patterns

Binary flags for major formations.

### ✔ 5. Add Price Structure

Swing highs/lows + structural trend classes (HH/HL/LH/LL).

### ✔ 6. Add Targets
	•	future_ret_1d, target_up_1d
	•	future_ret_5d, target_up_5d

### ✔ 7. Save Final Dataset

Output saved as:

```python
data/ohlcv_ml_ready.parquet
```

---

## ▶️ How to Run the Pipeline

### 1. Install dependencies:
   ```bash
   ppip install jugaad-data pandas numpy pyarrow tqdm
   ```

### 2. Configure Symbols & Dates

In the notebook:
```python
SYMBOLS = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
START = dt.date(2015, 1, 1)
END   = dt.date(2025, 1, 1)
```

### 3. Run All Cells

This will generate the full ML-ready dataset.

### 4. Verify Output

```python
df = pd.read_parquet("data/ohlcv_ml_ready.parquet")
df.head()
```
---

## 🚀 Roadmap (Next Steps)

### Phase 1 — Validation & EDA
	•	Check missing values
	•	Class imbalance analysis (1d vs 5d)
	•	Visualize structure labels & indicator trends

### Phase 2 — Tabular Baseline
	•	Train LightGBM / XGBoost / MLP models
	•	Compare target_up_1d vs target_up_5d

### Phase 3 — Candlestick Image Model
	•	Generate chart images (rolling windows)
	•	Train CNN (ResNet/EfficientNet)

### Phase 4 — Structure Sequence Model
	•	Build sequences of OHLCV + structure labels
	•	Train LSTM / GRU / Transformer

### Phase 5 — Multimodal Fusion
	•	Collect branch outputs:
p_up_tab, p_up_img, p_up_struct
	•	Train meta-classifier (MLP / ensemble / XGBoost)

### Phase 6 — Dissertation Writing
	•	Horizon comparison (1d vs 5d)
	•	Single vs multimodal performance
	•	Architecture diagrams
	•	Experiment results
	•	Final conclusions
