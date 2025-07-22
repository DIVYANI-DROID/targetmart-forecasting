# feature_engineering.py

import pandas as pd
import numpy as np

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Sort values (important for time series)
    df.sort_values(by=["product", "date"], inplace=True)
    
    # Lag feature: previous day’s sales
    df["sales_lag_1"] = df.groupby("product")["units_sold"].shift(1)
    
    # Rolling average features
    df["rolling_avg_3"] = df.groupby("product")["units_sold"].shift(1).rolling(window=3).mean().reset_index(0, drop=True)
    df["rolling_avg_7"] = df.groupby("product")["units_sold"].shift(1).rolling(window=7).mean().reset_index(0, drop=True)
    
    # Flag for promotion
    df["promo_flag"] = df["promo"].astype(int)
    
    # Date-related features
    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
    
    # Price change vs previous day
    df["price_change"] = df.groupby("product")["price"].pct_change()
    
    # Sales change vs previous day
    df["sales_change"] = df.groupby("product")["units_sold"].pct_change()
    
    # Price elasticity = % change in sales / % change in price
    df["price_elasticity"] = df["sales_change"] / df["price_change"]
    
    # Simple product encoding
    df["product_enc"] = df["product"].astype("category").cat.codes

    return df