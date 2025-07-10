# src/scripts/analysis/analyze_model.py

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path
import sys

# --- Add project root to the Python path ---
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from src.scripts.utils.config import load_config
from src.scripts.utils.logger import setup_logging, log_info, log_success, log_error

def analyze_feature_importance(model_path: str, plot_dir: str, top_n: int = 25):
    """
    Loads a trained model, extracts feature importances,
    and saves a plot of the most important features.
    """
    try:
        log_info(f"Loading model from {model_path}...")
        model = joblib.load(model_path)
    except FileNotFoundError:
        log_error(f"Model file not found at {model_path}. Please run the 'model' command first.")
        return

    # Check if the model has feature importances (e.g., tree-based models)
    if not hasattr(model, 'feature_importances_'):
        log_error(f"The model of type {type(model).__name__} does not support feature importances.")
        return

    # Create a DataFrame of features and their importance scores
    importances = pd.DataFrame({
        'feature': model.feature_names_in_,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    log_info("Top 10 Most Important Features:")
    print(importances.head(10).to_string(index=False))

    # --- Plotting ---
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(12, 10))

    # Plot only the top N features for clarity
    top_importances = importances.head(top_n)

    sns.barplot(x='importance', y='feature', data=top_importances, palette='viridis', ax=ax, hue='feature', legend=False)

    ax.set_title(f'Top {top_n} Feature Importances', fontsize=16)
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    # Ensure plot directory exists
    Path(plot_dir).mkdir(parents=True, exist_ok=True)
    output_path = Path(plot_dir) / 'feature_importance.png'
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    
    log_success(f"✅ Successfully saved feature importance plot to {output_path}")
    log_info("You can now view the plot to see what drives your model's predictions.")


def main():
    """
    Main CLI entrypoint for the model analysis script.
    """
    setup_logging()
    parser = argparse.ArgumentParser(description="Analyze a trained model's feature importances.")
    parser.add_argument("--config", default="config.yaml", help="Path to the project's config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config['data_paths']

    analyze_feature_importance(paths['model'], paths['plot_dir'])

if __name__ == "__main__":
    main()