🛍️ TargetMart – Personalized Promotions & Forecasting
A machine learning-powered dashboard for demand forecasting and product-level personalization, designed for retail and marketing teams to optimize promotions based on historical sales and calendar data.

<p align="center"> <img src="app/screenshot.png" alt="Dashboard Screenshot" width="80%"> </p>

📈 Key Features:

1. Forecasting Engine: Predicts product demand with improved accuracy (12% MAPE improvement).

2. Streamlit Dashboard: Interactive UI for exploring sales trends, pricing effects, and promo strategies.

3. Feature Engineering: Includes lag variables, moving averages, categorical flags, and price elasticity.

4. Personalization Layer: Tailors insights and recommendations per product and calendar event.

🧠 Tech Stack:

1. Python

2. Pandas, Scikit-learn, XGBoost

3. Hugging Face Transformers (for embedding logic if extended)

4. Streamlit for dashboard

6. Matplotlib, Seaborn for visualizations

7. Pickle for model storage

📁 Project Structure

targetmart-forecasting/
├── app/                     ← Streamlit UI
│   └── app.py
├── data/                    ← Sample sales dataset
│   └── sample_sales.csv
├── models/                  ← Saved ML models
│   └── best_rf_model.pkl, etc.
├── notebooks/               ← Data exploration and EDA
│   └── exploration.ipynb
├── src/
│   └── utils/               ← Feature engineering scripts
│       └── feature_engineering.py
├── requirements.txt
├── README.md
└── .gitignore

🚀 How to Run
1. Clone the repo

git clone https://github.com/your-username/targetmart-forecasting.git
cd targetmart-forecasting

2. Create a virtual environment

python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
Run the dashboard

bash
Copy
Edit
streamlit run app/app.py
📊 Sample Dashboard Screenshot
targetmart-forecasting/app/screenshot.jpeg
