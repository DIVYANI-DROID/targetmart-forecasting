# # Main Streamlit dashboard file

# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# import joblib
# from datetime import datetime
# import numpy as np
# from sklearn.metrics import mean_squared_error
# import plotly.express as px
# from utils.feature_engineering import engineer_features

# # === Load Data & Model ===
# @st.cache_data
# def load_and_engineer_data():
#     df = pd.read_csv("/Users/da/Documents/TARGET/targetmart-forecasting/data/sample_sales.csv", parse_dates=["date"])
#     df = engineer_features(df)
#     return df

# @st.cache_resource
# def load_model(product_name):
#     model_path = f"/Users/da/Documents/TARGET/targetmart-forecasting/models/best_rf_model_{product_name}.pkl"
#     return joblib.load(model_path)

# # === App Title ===
# st.title("🎯 TargetMart - Forecasting & Promotion Strategy Dashboard")
# st.markdown("Use this tool to forecast product demand, analyze promo impact, and optimize inventory.")
# # === Load the data ===
# df = load_and_engineer_data()


# # === Sidebar Filters ===
# product_list = df['product'].unique()
# selected_product = st.sidebar.selectbox("Select Product", product_list)
# date_range = st.sidebar.date_input("Select Date Range", [df["date"].min(), df["date"].max()])

# # === Load model aftre selecting the product ===
# model = load_model(selected_product)

# # === Filtered data ===
# filtered_df = df[(df['product'] == selected_product) & 
#                  (df['date'] >= pd.to_datetime(date_range[0])) & 
#                  (df['date'] <= pd.to_datetime(date_range[1]))]

# st.sidebar.markdown("### 🔧 Simulate Pricing & Promotion")
# simulated_price = st.sidebar.slider("Change Price (₹)", 10, 500, step=5, value=int(filtered_df["price"].mean()))
# simulated_promo = st.sidebar.slider("Simulate Promo Discount (%)", 0, 100, step=5, value=int(filtered_df["promo"].mean() * 100))

# X_sim = filtered_df.drop(columns=["date", "units_sold"]).copy()
# X_sim["price"] = simulated_price
# X_sim["promo"] = 1 if simulated_promo > 0 else 0

# # feature_cols = [
# #     "price", "promo", "promo_flag", "day_of_week", "is_weekend",
# #     "sales_lag_1", "rolling_avg_3", "rolling_avg_7",
# #     "price_change", "sales_change", "price_elasticity", "product_enc"
# # ]

# feature_cols = joblib.load("models/feature_cols.pkl")
# st.write("X_sim columns:", X_sim.columns.tolist())

# y_pred_sim = model.predict(X_sim[feature_cols])

# # Prepare simulation DataFrame with same feature engineering as training
# le = joblib.load("models/label_encoder_product.pkl")

# # Encode 'product' using same encoder as training
# X_sim["product_enc"] = le.transform(X_sim["product"])

# # Add engineered features — same as in training
# X_sim["day_of_week"] = X_sim["date"].dt.dayofweek
# X_sim["is_weekend"] = X_sim["day_of_week"].isin([5, 6]).astype(int)

# # Fill lag features and others as needed
# X_sim = X_sim.sort_values("date")
# X_sim["lag_1"] = X_sim["units_sold"].shift(1)
# X_sim["rolling_mean_3"] = X_sim["units_sold"].rolling(window=3).mean()

# X_sim["price_change"] = X_sim["price"].pct_change().fillna(0)
# X_sim["price_elasticity"] = (
#     X_sim["units_sold"].pct_change() / X_sim["price_change"]
# ).replace([np.inf, -np.inf], 0).fillna(0)

# # Ensure only expected columns are used
# X_sim = X_sim[feature_cols]

# y_pred_sim = model.predict(X_sim)

# # === Display Raw Data ===
# with st.expander("📄 Show Filtered Data"):
#     st.dataframe(filtered_df)

# # === Forecast Plot ===
# st.subheader("📈 Forecast Plot")

# plot_df = filtered_df.copy()
# plot_df["Predicted Units Sold"] = y_pred_sim

# fig = px.line(
#     plot_df,
#     x="date",
#     y=["units_sold", "Predicted Units Sold"],
#     labels={"value": "Units Sold", "date": "Date", "variable": "Legend"},
#     title=f"📊 Forecast vs Actual for {selected_product}"
# )
# st.plotly_chart(fig, use_container_width=True)

# # fig, ax = plt.subplots()
# # sns.lineplot(data=filtered_df, x="date", y="units_sold", label="Actual", ax=ax)

# # === (Placeholder for Predictions) ===
# # Add your model predictions here
# # pred = model.predict(...)

# # ax.plot(filtered_df["date"], pred, label="Forecast", linestyle="--")
# # ax.legend()

# # st.pyplot(fig)

# # === Feature Importance ===
# st.subheader("🧠 Feature Importance")

# # Placeholder example
# feature_importance = pd.DataFrame({
#     'Feature': ['lag_1', 'rolling_avg_3', 'price', 'promo'],
#     'Importance': [0.30, 0.25, 0.20, 0.15]
# })

# fig2, ax2 = plt.subplots()
# sns.barplot(data=feature_importance, x='Importance', y='Feature', ax=ax2)
# st.pyplot(fig2)

# # === MAPE Score ===
# st.metric("📊 MAPE Score", "12%")  # Replace with dynamic value if available
# st.metric("Random Forest RMSE", "2.15")

# # Model Feedback
# st.subheader("📝 Model Feedback")

# st.write(f"**Model Used:** Random Forest Regressor")
# st.write(f"**Forecast Date Range:** {date_range[0]} to {date_range[1]}")
# st.write("**Assumptions:** Promo and Price are treated as fixed inputs from historical data.")

# # === Promotions & Price Sensitivity ===
# st.subheader("📅 Promo & Pricing Strategy")

# fig3, ax3 = plt.subplots()
# sns.lineplot(data=filtered_df, x="date", y="units_sold", label="Units Sold", ax=ax3)
# sns.scatterplot(data=filtered_df[filtered_df['promo'] == 1], x="date", y="units_sold", color="red", label="Promo", ax=ax3)
# ax3.set_title("Promotional Impact")
# st.pyplot(fig3)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.feature_engineering import engineer_features

# === Load models & encoders ===
model_dir = "/Users/da/Documents/TARGET/targetmart-forecasting/models"

@st.cache_resource
def load_models():
    models = {}
    for file in os.listdir(model_dir):
        if file.startswith("best_rf_model_"):
            product = file.replace("best_rf_model_", "").replace(".pkl", "")
            models[product] = joblib.load(os.path.join(model_dir, file))
    return models

@st.cache_resource
def load_encoder_and_features():
    le = joblib.load(f"{model_dir}/label_encoder_product.pkl")
    feature_cols = joblib.load(f"{model_dir}/feature_cols.pkl")
    return le, feature_cols

models = load_models()
le, feature_cols = load_encoder_and_features()

# === Load & engineer data ===
@st.cache_data
def load_and_engineer_data():
    df = pd.read_csv("/Users/da/Documents/TARGET/targetmart-forecasting/data/sample_sales.csv", parse_dates=["date"])
    df = engineer_features(df)
    return df

df = load_and_engineer_data()

# === Streamlit UI ===
st.set_page_config(layout="wide")
st.title("📈 TargetMart Forecasting Dashboard")

# === Sidebar filters ===
product_choices = sorted(df["product"].unique())
selected_product = st.sidebar.selectbox("Select Product", product_choices)

# === Filtered Data ===
df_product = df[df["product"] == selected_product].sort_values("date")

# === Model prediction ===
model = models.get(selected_product)

if model:
    X = df_product[feature_cols]
    # Replace inf with nan
    X.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop or fill NaNs (choose one)

    ## Option 1: Fill with a safe value (e.g., 0)
    X.fillna(0, inplace=True)

    ## Option 2: Drop rows with NaNs
    # X.dropna(inplace=True)
    print("Any NaNs?", np.isnan(X).any())
    print("Any Infs?", np.isinf(X).any())
    print("Max value:", np.max(X.values))
    print("Min value:", np.min(X.values))
    y = df_product["units_sold"]
    y_pred = model.predict(X)
    
    df_product["forecast"] = y_pred

    # === Plot ===
    fig = px.line(
        df_product,
        x="date",
        y=["units_sold", "forecast"],
        labels={"value": "Units Sold", "variable": "Legend"},
        title=f"📦 Sales Forecast for {selected_product}"
    )
    st.plotly_chart(fig, use_container_width=True)

    # === Metrics ===
    rmse = np.sqrt(np.mean((y - y_pred) ** 2))
    mape = np.mean(np.abs((y - y_pred) / y)) * 100

    col1, col2 = st.columns(2)
    col1.metric("RMSE", f"{rmse:.2f}")
    col2.metric("MAPE", f"{mape:.2f}%")

else:
    st.warning(f"No model found for {selected_product}.")

# === Simulate Price Change ===
st.subheader("🛠 Simulate Promotion or Price Change")

with st.form("simulation_form"):
    price_change = st.slider("Change Price by (%)", -50, 50, 0)
    promo_flag = st.selectbox("Promotion Applied?", ["Yes", "No"]) == "Yes"
    submit = st.form_submit_button("Predict Impact")

if submit:
    # Clone latest record and simulate changes
    latest = df_product.sort_values("date").iloc[-1:].copy()
    latest["date"] = latest["date"] + pd.Timedelta(days=1)
    latest["price"] *= 1 + (price_change / 100)
    latest["promo"] = promo_flag

    # Recompute engineered features
    X_sim = pd.concat([df_product, latest])
    X_sim = engineer_features(X_sim)
    X_sim = X_sim.sort_values("date").reset_index(drop=True)

    latest_row = X_sim.iloc[[-1]]
    y_pred_sim = model.predict(latest_row[feature_cols])[0]

    st.success(f"📊 Predicted Units Sold (Tomorrow): {y_pred_sim:.2f}")
