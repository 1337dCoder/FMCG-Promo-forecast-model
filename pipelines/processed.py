import pandas as pd
import numpy as np
import os

def load_raw_data():
    """Load raw Rossmann CSV files."""
    train = pd.read_csv('data/raw/train.csv', low_memory=False)
    store = pd.read_csv('data/raw/store.csv')
    print(f"Train loaded: {train.shape}")
    print(f"Store loaded: {store.shape}")
    return train, store


def merge_data(train, store):
    """Merge store metadata into train data."""
    df = train.merge(store, on='Store', how='left')
    print(f"Merged shape: {df.shape}")
    return df


def filter_bad_rows(df):
    """Remove closed stores and zero sales rows."""
    before = len(df)
    df = df[df['Open'] == 1]
    df = df[df['Sales'] > 0]
    after = len(df)
    print(f"Rows removed: {before - after:,}")
    print(f"Rows remaining: {after:,}")
    return df


def fix_data_types(df):
    """Fix column data types."""
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfYear'] = df['Date'].dt.dayofyear
    print("Date column parsed and time features extracted")
    return df


def handle_missing_values(df):
    """Handle missing values with business logic."""
    df['CompetitionDistance'] = df['CompetitionDistance'].fillna(
        df['CompetitionDistance'].median()
    )
    df['CompetitionOpenSinceMonth'] = df['CompetitionOpenSinceMonth'].fillna(0)
    df['CompetitionOpenSinceYear'] = df['CompetitionOpenSinceYear'].fillna(0)
    df['Promo2SinceWeek'] = df['Promo2SinceWeek'].fillna(0)
    df['Promo2SinceYear'] = df['Promo2SinceYear'].fillna(0)
    df['PromoInterval'] = df['PromoInterval'].fillna('None')
    missing = df.isnull().sum().sum()
    print(f"Total missing values remaining: {missing}")
    return df


def encode_categoricals(df):
    """Encode categorical columns as numbers."""
    df['StoreType'] = df['StoreType'].map(
        {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    )
    df['Assortment'] = df['Assortment'].map(
        {'a': 0, 'b': 1, 'c': 2}
    )
    df['StateHoliday'] = df['StateHoliday'].map(
        {'0': 0, 0: 0, 'a': 1, 'b': 1, 'c': 1}
    )
    print("Categorical columns encoded")
    return df


def save_processed_data(df):
    """Save clean data to processed folder."""
    os.makedirs('data/processed', exist_ok=True)
    output_path = 'data/processed/train_clean.csv'
    df.to_csv(output_path, index=False)
    print(f"Clean data saved to {output_path}")
    print(f"Final shape: {df.shape}")


def run_pipeline():
    """Run the full preprocessing pipeline."""
    print("="*50)
    print("STARTING PREPROCESSING PIPELINE")
    print("="*50)
    train, store = load_raw_data()
    df = merge_data(train, store)
    df = filter_bad_rows(df)
    df = fix_data_types(df)
    df = handle_missing_values(df)
    df = encode_categoricals(df)
    save_processed_data(df)
    print("="*50)
    print("PIPELINE COMPLETE")
    print("="*50)
    return df


if __name__ == "__main__":
    df = run_pipeline()