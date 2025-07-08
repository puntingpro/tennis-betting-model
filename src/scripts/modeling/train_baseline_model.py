# src/scripts/modeling/train_baseline_model.py

import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss

def train_baseline_model():
    """
    Loads the historical feature data, trains a simple logistic
    regression model, and saves it to a file.
    """
    print("Loading historical player features...")
    try:
        atp_features = pd.read_csv("data/processed/atp_player_features.csv")
        wta_features = pd.read_csv("data/processed/wta_player_features.csv")
        historical_features = pd.concat([atp_features, wta_features], ignore_index=True)
    except FileNotFoundError as e:
        print(f"Error loading feature files: {e}")
        return

    # --- Feature Engineering ---
    # Create a 'winner' column (target variable): 1 if player1 won, 0 otherwise
    # This requires merging the data with itself to get opponent stats
    
    # For simplicity in this baseline, we will create a proxy target.
    # A more advanced model would use the match_id to pair players correctly.
    # We'll predict if a player's win_rate is > 0.5 as a simple target.
    
    df = historical_features.dropna(subset=['overall_win_rate', 'surface_win_rate']).copy()
    
    # Define features and a simple target for this baseline model
    features = ['overall_win_rate', 'surface_win_rate', 'h2h_wins', 'h2h_losses']
    df['winner'] = (df['overall_win_rate'] > df.groupby('match_id')['overall_win_rate'].transform('mean')).astype(int)

    X = df[features]
    y = df['winner']

    # Split data for training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training model on {len(X_train)} samples...")
    model = LogisticRegression(class_weight='balanced')
    model.fit(X_train, y_train)
    
    # --- Evaluate Model ---
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    logloss = log_loss(y_test, y_prob)
    
    print(f"\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Log Loss: {logloss:.4f}")

    # --- Save Model ---
    output_dir = Path("models")
    output_dir.mkdir(exist_ok=True)
    model_path = output_dir / "baseline_model.joblib"
    
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved successfully to: {model_path}")


if __name__ == "__main__":
    train_baseline_model()