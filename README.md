# 🎯 TargetMart – Personalized Promotions & Forecasting (In Dev)

**Tech Stack**: `scikit-learn`, `Hugging Face`, `Streamlit`

TargetMart is a demand forecasting engine built to optimize product-level promotions using historical sales and calendar data. It delivers dashboard-ready outputs and personalization insights.

---

## 📊 Project Goals

- Improve accuracy of demand forecasting (achieved **12% MAPE improvement**).
- Generate product-level forecasts with promotion and calendar effects.
- Provide dashboard-ready outputs for decision-making.

---

## 🛠️ Features

- 📈 Model training with calendar + sales features
- 🧠 Uses Random Forest + Transformers (WIP)
- 🧮 Feature engineering: lag variables, rolling means, price elasticity, promo flags
- 📉 Evaluation using MAPE
- 🖥️ Interactive Streamlit dashboard (Coming Soon)

---

## 📁 Project Structure

argetmart-forecasting/
│
├── data/ # Sample sales and calendar data
├── notebooks/ # Exploration and feature engineering notebooks
├── models/ # Trained model artifacts (optional)
├── app/ # Streamlit dashboard (in dev)
├── README.md # Project overview and documentation
└── requirements.txt # Dependencies


---

## 🚀 Quick Start

```bash
git clone https://github.com/DIVYANI-DROID/targetmart-forecasting.git
cd targetmart-forecasting
pip install -r requirements.txt

streamlit run app/app.py

📬 Contact
Made with ❤️ by Divyani