import pandas as pd
import numpy as np
import os


def load_clean_data():
    """Load the processed data."""
    df = pd.read_csv('data/processed/train_clean.csv', low_memory=False)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(['Store', 'Date']).reset_index(drop=True)
    print(f"Loaded clean data: {df.shape}")
    return df


def add_time_features(df):
    """Add calendar-based features."""
    df['IsWeekend'] = df['DayOfWeek'].isin([6, 7]).astype(int)
    df['IsMonthStart'] = (df['Day'] <= 3).astype(int)
    df['IsMonthEnd'] = (df['Day'] >= 28).astype(int)
    df['Quarter'] = df['Date'].dt.quarter
    print("Time features added")
    return df


def add_lag_features(df):
    """Add lag features per store."""
    print("Adding lag features...")
    
    for lag in [7, 14, 28]:
        df[f'Sales_Lag_{lag}'] = df.groupby('Store')['Sales'].transform(
            lambda x: x.shift(lag)
        )
        print(f"  Lag {lag} added")
    
    return df


def add_rolling_features(df):
    """Add rolling average features per store."""
    print("Adding rolling features...")
    
    for window in [7, 14, 28]:
        df[f'Sales_Rolling_Mean_{window}'] = df.groupby('Store')['Sales'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f'Sales_Rolling_Std_{window}'] = df.groupby('Store')['Sales'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )
        print(f"  Rolling {window} added")
    
    return df


def add_promo_features(df):
    """Add promotion-specific features."""
    
    # How many days has promo been running consecutively?
    df['PromoRunLength'] = df.groupby('Store')['Promo'].transform(
        lambda x: x.groupby((x != x.shift()).cumsum()).cumcount() + 1
    ) * df['Promo']
    
    # Was there a promo yesterday?
    df['PromoYesterday'] = df.groupby('Store')['Promo'].transform(
        lambda x: x.shift(1)
    ).fillna(0)
    
    print("Promo features added")
    return df


def add_competition_features(df):
    """Add competition-based features."""
    
    # How long has competitor been open (in months)?
    df['CompetitionOpenMonths'] = (
        (df['Year'] - df['CompetitionOpenSinceYear']) * 12 +
        (df['Month'] - df['CompetitionOpenSinceMonth'])
    ).clip(lower=0)
    
    # Replace invalid values
    df['CompetitionOpenMonths'] = df['CompetitionOpenMonths'].fillna(0)
    
    print("Competition features added")
    return df


def drop_nulls_from_lags(df):
    """Drop rows where lag features are null."""
    before = len(df)
    df = df.dropna(subset=['Sales_Lag_7', 'Sales_Lag_14', 'Sales_Lag_28'])
    after = len(df)
    print(f"Rows dropped due to lag nulls: {before - after:,}")
    print(f"Rows remaining: {after:,}")
    return df


def save_features(df):
    """Save feature-engineered dataset."""
    output_path = 'data/processed/train_features.csv'
    df.to_csv(output_path, index=False)
    print(f"Features saved to {output_path}")
    print(f"Final shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")


def run_feature_pipeline():
    """Run the full feature engineering pipeline."""
    print("="*50)
    print("STARTING FEATURE ENGINEERING PIPELINE")
    print("="*50)
    
    df = load_clean_data()
    df = add_time_features(df)
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = add_promo_features(df)
    df = add_competition_features(df)
    df = drop_nulls_from_lags(df)
    save_features(df)
    
    print("="*50)
    print("FEATURE PIPELINE COMPLETE")
    print("="*50)
    return df


if __name__ == "__main__":
    df = run_feature_pipeline()