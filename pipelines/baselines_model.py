import pandas as pd
import numpy as np

def mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def run_baseline():
    print("="*50)
    print("BASELINE MODEL")
    print("="*50)

    df = pd.read_csv('data/processed/train_features.csv')

    # Temporal split - last 6 weeks as validation
    df['Date'] = pd.to_datetime(df['Date'])
    split_date = df['Date'].max() - pd.Timedelta(weeks=6)

    train = df[df['Date'] <= split_date]
    val = df[df['Date'] > split_date]

    print(f"Train size: {len(train):,}")
    print(f"Validation size: {len(val):,}")
    print(f"Split date: {split_date.date()}")

    # Baseline: mean sales per store per day of week
    store_dow_mean = train.groupby(
        ['Store', 'DayOfWeek']
    )['Sales'].mean()

    val = val.copy()
    val['Predicted'] = val.set_index(
        ['Store', 'DayOfWeek']
    ).index.map(store_dow_mean)

    # Fill any missing with global mean
    global_mean = train['Sales'].mean()
    val['Predicted'] = val['Predicted'].fillna(global_mean)

    # Evaluate
    mape = mean_absolute_percentage_error(
        val['Sales'].values,
        val['Predicted'].values
    )
    rmse = np.sqrt(np.mean((val['Sales'] - val['Predicted'])**2))

    print(f"\nBaseline MAPE: {mape:.2f}%")
    print(f"Baseline RMSE: {rmse:.2f}")
    print("\nThe basic XGBoost should beat.")
    print("="*50)

if __name__ == "__main__":
    run_baseline()