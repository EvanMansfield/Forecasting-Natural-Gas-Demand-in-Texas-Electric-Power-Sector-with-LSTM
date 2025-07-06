Forecasting Natural Gas Demand in Texas’ Electric Power Sector with LSTM

📑 Assignment

This repository contains the final project for Data 6575: Deep Learning & AI. The deliverable is a Jupyter notebook—Data6575_FinalProject_EvanMansfield.ipynb—which builds, trains, and evaluates recurrent neural-network models to forecast monthly natural-gas demand in the Texas electric-power sector one month ahead.

🎯 Objective

Accurate short-term gas-demand forecasts allow utilities and commodity desks to lock in supply and hedge price exposure in ERCOT’s highly volatile market. The goal is to beat a naïve “same-as-last-month” benchmark while remaining simple enough for real-world deployment.

🗂️ Repository structure

.
├── Data6575_FinalProject_EvanMansfield.ipynb   # Main notebook (analysis & results)
├── data/                                      # Raw & processed CSVs (ignored – see Data section)
├── models/                                    # Saved PyTorch checkpoints
├── figures/                                   # Plots exported by the notebook
└── README.md

The data/, models/, and figures/ folders are created automatically the first time you run the notebook and are git-ignored to keep the repo lightweight.

🏛️ Data sources

Dataset	Description	Frequency	Access
EIA-923 Form	Monthly natural-gas consumption (MWh equivalent) by plant & state	Monthly	Bulk CSV download from eia.gov
NOAA GHCN-Daily	Daily Tmax/Tmin for six major Texas stations (converted to heating & cooling degree-days)	Daily	Public FTP

The notebook downloads and caches the raw files automatically—no manual action required.

🛠️ Environment & installation

# 1. Clone the repo
git clone https://github.com/<your-user>/<your-repo>.git
cd <your-repo>

# 2. Create & activate virtual env (recommended)
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

If you prefer Conda, an environment.yml snippet is provided in the notebook header.

Key libraries
	•	pandas, numpy
	•	torch (CPU-only is fine)
	•	scikit-learn
	•	matplotlib

 Running the notebook
	1.	Launch Jupyter: jupyter lab (or jupyter notebook).
	2.	Open Data6575_FinalProject_EvanMansfield.ipynb.
	3.	Run Kernel → Restart & Run All (≈ 5 min on an M1 MacBook Air; GPU optional).

Intermediate artefacts—feature-engineered CSVs, degree-day caches, model checkpoints—are written to /data and /models, so subsequent runs are instant.

Methodology

Step	Details
Feature engineering	Aggregate plant-level gas usage → state level; derive monthly HDD/CDD from station weather; standardise features.
Windowing	Sliding window of 12 past months → next-month target (custom TensorDataset).
Models	1) Baseline unidirectional LSTM (1 layer, hidden = 32); 2) Enhanced bidirectional LSTM with LayerNorm & Dropout.
Training	AdamW optimiser, One-Cycle LR scheduler, 100 epochs, early-stopping on validation MAE.
Evaluation	MAE & MAPE on a 20-month hold-out set. Baseline ≈ 8,800 MWh MAE; Bi-LSTM over-fits the small dataset.

Results

The baseline LSTM captures seasonal swings and outperforms naïve persistence by ~25 %. Adding bidirectional context improved training loss but did not generalise as well.

⸻

Author – Evan Mansfield · July 2025
