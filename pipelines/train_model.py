import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import os
from baselines_model import mean_absolute_percentage_error

def get_features():
    #Define which columns to use as features.
    drop_cols = ['Sales', 'Date', 'Customers', 'Open', 
                 'PromoInterval', 'CompetitionOpenSinceMonth',
                 'CompetitionOpenSinceYear', 'Promo2SinceWeek',
                 'Promo2SinceYear']
    return drop_cols

def run_training():
    print("="*50)
    print("XGBOOST TRAINING PIPELINE")
    print("="*50)

    # Load data
    df = pd.read_csv('data/processed/train_features.csv')
    df['Date'] = pd.to_datetime(df['Date'])

    # Temporal split
    split_date = df['Date'].max() - pd.Timedelta(weeks=6)
    train = df[df['Date'] <= split_date].copy()
    val = df[df['Date'] > split_date].copy()

    print(f"Train size: {len(train):,}")
    print(f"Val size: {len(val):,}")

    # Define features
    drop_cols = get_features()
    feature_cols = [c for c in df.columns if c not in drop_cols]
    
    print(f"Number of features: {len(feature_cols)}")

    X_train = train[feature_cols]
    y_train = train['Sales']
    X_val = val[feature_cols]
    y_val = val['Sales']

    # Train XGBoost
    print("\nTraining XGBoost...")
    model = xgb.XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
        eval_metric='rmse'
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )

    # Evaluate
    val_preds = model.predict(X_val)
    val_preds = np.maximum(val_preds, 0)

    mape = mean_absolute_percentage_error(y_val.values, val_preds)
    rmse = np.sqrt(np.mean((y_val.values - val_preds)**2))

    print(f"\nXGBoost MAPE: {mape:.2f}%")
    print(f"XGBoost RMSE: {rmse:.2f}")
    print(f"\nBaseline MAPE was: 18.55%")
    print(f"Baseline RMSE was: 1658.79")
    print(f"Improvement: {18.55 - mape:.2f}% MAPE reduction")

    # Save model
    os.makedirs('models', exist_ok=True)
    with open('models/xgboost_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Save feature list
    with open('models/feature_cols.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)

    print("\nModel saved to models/xgboost_model.pkl")
    print("="*50)

if __name__ == "__main__":
    run_training()