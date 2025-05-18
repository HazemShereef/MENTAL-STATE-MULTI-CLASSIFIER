# 🧠 Mental State Multi-Classifier  

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)  
*A real-time CNN classifier that detects mental states (Baseline, Stress, Amusement, Meditation) from physiological signals (ECG, EDA, Respiration).*

---

## 📌 **Key Features**  
- **4-class classification** (93-98% accuracy)  
- **Optimized for wearables**: Uses only 3 biosignals  
- **5-second real-time prediction** (3500 samples at 700Hz)  
- **Robust preprocessing**: SMOTE + Gaussian noise augmentation  
- **Lightweight CNN**: Trains in <10 mins on consumer hardware  

---

## 🛠 **Tech Stack**  
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![NeuroKit2](https://img.shields.io/badge/NeuroKit2-0.2.1-green)

**Libraries**:  
- Signal processing: `NeuroKit2`  
- ML: `PyTorch`, `scikit-learn`  
- Visualization: `TensorBoard`  

---

## 📂 **Dataset**  
**WESAD** (Wearable Stress and Affect Detection):  
- 15 subjects × 4 states  
- **Signals**: Chest-worn RespiBAN (ECG/EDA/Respiration)  
- **Sample rate**: 700Hz  

---

## ⚙️ **Preprocessing Pipeline**  
1. **Filtering**  
   - ECG: FIR [0.67-45Hz] + 50Hz notch  
   - EDA: Low-pass [3Hz]  
   - Respiration: Bandpass [0.05-0.5Hz]  

2. **Segmentation** → 5-second windows  

3. **Augmentation**  
   ```python
   # SMOTE for class balancing
   from imblearn.over_sampling import SMOTE
   sm = SMOTE(random_state=42)
   X_res, y_res = sm.fit_resample(X_train, y_train)
   
