# Wavelet Deep Learning for Multi-Time Scale Affect Forecasting  

This repository contains code for the paper:  
*"Interpreting Feature Importance in Wavelet-Based Deep Learning for Multi-Time Scale Affect Forecasting"*

## 📂 Repository Structure  
```bash
📂 wavelet-affect/
│── 📂 empirical/        # Empirical example
│   │── 📂 data/         # Sample dataset (full dataset available separately)
│   │── Empirical_ADID_ScatteringDemo.ipynb
│
│── 📂 simulation/       # Simulation example 
│   │── Simul_Illus1-2GeneratePlotExamples.ipynb
│   │── Simul_Illus3TimeVaryingFreqDemo.ipynb
│   │── myTVFunctions.py
```

## 🚀 Getting Started  
### 1️⃣ Clone the Repository
First, download the repository to your local machine:
```sh
git clone https://github.com/Young1Cho/wavelet-affect.git
cd wavelet-affect
```
### 2️⃣ Install Dependencies  
```sh
pip install -r requirements.txt
```

### 3️⃣ Download Dataset
- **Full dataset** (Access required): [Penn State SharePoint](https://pennstateoffice365.sharepoint.com/:f:/s/EPiC2/EmBDPx0ir5xNmdOsToX1iYgBWj0wTgG-9rfQeiUO5Xvsyg?e=DOec6s)
- **Sample dataset** (Public; already available in the repository)

After downloading the full dataset, place it in the `empirical/data/` folder:
```bash
📂 wavelet-affect/
│── 📂 empirical/        # Empirical example
│   │── 📂 data/         # Sample dataset (full dataset available separately)
│   │   │── AllSlide50msec.csv  # full_dataset
│   │   │── sample_data.csv     # (Already included in the repository)
```

### 4️⃣ Run the Code (Work in progress)
🔹 Run Main Script
```sh
python src/main.py
```
🔹 Run Jupyter Notebook (If Using Notebooks)
```sh
jupyter notebook
```

<!-- ##
### Citing This Work
If this code is used in research, please cite:
```bibtex
@inproceedings{chow2025interpreting,
  author    = {Sy-Miin Chow and Young Won Cho and Xiaoyue Xiong and Yanling Li and Yuqi Shen and Jyotirmoy Das and Linying Ji and Soundar Kumara},
  title     = {Interpreting Feature Importance in Wavelet-Based Deep Learning for Multi-Time Scale Affect Forecasting},
  booktitle = {Proceedings of the 90th Annual International Meeting of the Psychometric Society},
  year      = {2025},
  address   = {Minneapolis, United States}
}
``` -->
