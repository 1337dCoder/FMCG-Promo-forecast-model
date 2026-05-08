import pandas as pd
import numpy as np
import shap
import pickle
import matplotlib.pyplot as plt
import os

def run_explainability():
    print("="*50)
    print("SHAP EXPLAINABILITY")
    print("="*50)

    # Load model and features
    with open('models/xgboost_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('models/feature_cols.pkl', 'rb') as f:
        feature_cols = pickle.load(f)

    # Load sample of validation data
    df = pd.read_csv('data/processed/train_features.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    split_date = df['Date'].max() - pd.Timedelta(weeks=6)
    val = df[df['Date'] > split_date].copy()

    # Sample 500 rows for speed
    sample = val[feature_cols].sample(500, random_state=42)

    print("Calculating SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # Save summary plot
    os.makedirs('reports', exist_ok=True)
    plt.figure()
    shap.summary_plot(shap_values, sample, show=False)
    plt.tight_layout()
    plt.savefig('reports/shap_summary.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("SHAP summary plot saved to reports/shap_summary.png")

    # Print top 10 most important features
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'MeanAbsSHAP': np.abs(shap_values).mean(axis=0)
    }).sort_values('MeanAbsSHAP', ascending=False)

    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10).to_string(index=False))
    print("="*50)

if __name__ == "__main__":
    run_explainability()