# 🛒 AI-Driven Retail Demand Forecasting & Dynamic Pricing System

<p>
  <img src="https://img.shields.io/badge/Language-Python-3776AB?logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Deep%20Learning-PyTorch-EF4B4B?logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Time%20Series-TFT-blue"/>
  <img src="https://img.shields.io/badge/Reinforcement%20Learning-PPO-green"/>
  <img src="https://img.shields.io/badge/Forecasting-PyTorch%20Forecasting-orange"/>
  <img src="https://img.shields.io/badge/Training-Lightning-purple"/>
  <img src="https://img.shields.io/badge/Explainability-SHAP-yellow"/>
  <img src="https://img.shields.io/badge/Visualization-Matplotlib-11557C"/>
</p>

---

## 🚀 Building an Intelligent Retail System for Demand Forecasting and Dynamic Pricing

This project presents an end-to-end AI system that combines deep learning and reinforcement learning to address two key retail challenges:

- 📈 Demand forecasting using Temporal Fusion Transformer (TFT)  
- 💰 Dynamic pricing optimization using Proximal Policy Optimization (PPO)  

Unlike traditional forecasting-only approaches, this system integrates prediction and decision-making, making it closer to real-world retail AI applications used in companies like Amazon and Walmart.

---

## 🎯 Why This Project Matters

Retail businesses constantly struggle with:

- Uncertain customer demand  
- Overstocking and stockouts  
- Static pricing strategies  

This project demonstrates how AI can:

- Predict future demand patterns  
- Adapt prices dynamically based on demand  
- Optimize revenue using learning-based decision systems  

---

## 🏗️ Architecture Overview

```text
Raw Sales Data
      ↓
Data Processing & Feature Engineering
      ↓
Temporal Fusion Transformer (TFT)
      ↓
Demand Forecasts
      ↓
Reinforcement Learning (PPO Agent)
      ↓
Dynamic Pricing Decisions
```
---

## 🚀 Key Highlights

- 📈 Forecasted retail demand using Temporal Fusion Transformer (TFT)
- 💰 Optimized pricing using Reinforcement Learning (PPO)
- 📊 Achieved SMAPE of **1.532** on multi-step forecasting
- 🧠 Built explainable AI pipeline using SHAP and attention mechanisms
- 🔄 Combined prediction and decision-making in a single system

---

## 📊 Project Snapshot

- **Dataset:** Walmart M5 Forecasting Dataset  
- **Approach:** Deep Learning + Reinforcement Learning  
- **Forecast Horizon:** 7 Days  
- **Models Used:**  
  - Temporal Fusion Transformer (Demand Forecasting)  
  - PPO (Dynamic Pricing Optimization)

---

## ⚙️ System Design

### 1️⃣ Data Preparation
- Converted wide-format data into time-series format  
- Filtered subset for efficient training (1 store, top 30 items, last 180 days)  
- Merged sales, calendar, and pricing data  

---

### 2️⃣ Feature Engineering

- Lag features: `lag_1`, `lag_7`, `lag_14`  
- Rolling averages: `rolling_mean_7`, `rolling_mean_14`  
- Time-based features: `day_of_week`, `month`, `year`, `week_of_year`, `is_weekend`  
- Created sequential time index for modeling

---

### 3️⃣ Demand Forecasting (TFT)

- Multi-step forecasting model  
- Captures temporal dependencies and seasonality  
- Uses attention for interpretability  

#### 📈 Performance Metrics

- **SMAPE:** 1.532  
- **MAE:** 2.110  
- **RMSE:** 3.796  

📌 Interpretation: The model captures overall demand patterns and seasonality but shows moderate performance due to sparsity and volatility in retail sales data.

---

### 4️⃣ Model Explainability

- SHAP analysis for feature importance  
- TFT attention for time-step importance
- Variable importance plots

📌 Insight: Rolling means and lag features are strongest predictors of demand  

---

### 5️⃣ Dynamic Pricing Optimization (PPO)

- Custom Pricing Environment  
- PPO agent learns pricing strategy  
- Reward based on profit  

#### 📊 Training Results

- 10,000+ timesteps  
- Increasing cumulative reward  
- Improved explained variance  
- Better pricing decisions over time  

📌 Interpretation: The agent learns progressively better pricing strategies, as indicated by increasing cumulative reward and improved explained variance during training.

---

## 📊 Analysis & Insights

- Demand trends over time  
- Forecast vs actual comparison  
- SHAP feature importance  
- PPO reward growth  
- Price vs demand interaction  

---

## 🧠 Tech Stack

- Python  
- Pandas, NumPy  
- PyTorch  
- PyTorch Forecasting  
- PyTorch Lightning  
- Stable-Baselines3  
- Gym  
- SHAP  
- Matplotlib  

---

## 📂 Repository Structure

```
retail-ai-demand-pricing/
│
├── README.md
├── retail-demand-forecasting-dynamic-pricing.ipynb
├── retail-ai-project-report.pdf
├── retail-ai-project-presentation.pdf
│
└── data/
    └── dataset_link.txt
```

---

## ▶️ How to Run

1. Clone the repository  
2. Download dataset from Kaggle  
3. Update dataset path in notebook
4. Install dependencies  

```
pip install pandas numpy torch pytorch-lightning pytorch-forecasting stable-baselines3 gym shap matplotlib
```

5. Run notebook  

```
jupyter notebook retail-demand-forecasting-dynamic-pricing.ipynb
```

---

## 💼 Business Impact

- Enables data-driven demand forecasting for inventory planning  
- Demonstrates adaptive pricing strategies using reinforcement learning  
- Reduces reliance on static pricing models  
- Provides interpretable insights for business decision-making  

---

## ⚠️ Limitations

- Uses subset of dataset  
- Moderate forecasting performance  
- Pricing evaluated in simulation  

---

## 🔮 Future Improvements

- Scale to full dataset  
- Add promotions and external factors  
- Real-time deployment  
- Advanced RL models  

---

## ✅ Conclusion

This project presents an end-to-end AI system that integrates demand forecasting and dynamic pricing within a retail context. By combining Temporal Fusion Transformer (TFT) for multi-step forecasting with reinforcement learning (PPO) for pricing optimization, the system demonstrates how prediction and decision-making can be unified into a single workflow.

The results highlight that while forecasting performance is moderate due to the inherent variability of retail data, the overall pipeline successfully captures demand patterns and learns adaptive pricing strategies over time.

This project illustrates how combining forecasting and decision-making can lead to more practical and impactful AI solutions for real-world retail systems.


