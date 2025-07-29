import pandas as pd
from pathlib import Path
import sys

# Add the project root to the Python path to allow for module imports
sys.path.append(str(Path(__file__).resolve().parents[1]))

from tennis_betting_model.utils.config import load_config
from tennis_betting_model.utils.logger import (
    setup_logging,
    log_info,
    log_success,
    log_error,
)


def inspect_feature_file():
    """
    Loads the final features CSV and prints a summary to verify its contents,
    with a focus on the 'surface' column.
    """
    setup_logging()

    try:
        config = load_config("config.yaml")
        paths = config["data_paths"]
        features_path = Path(paths["consolidated_features"])
    except (FileNotFoundError, KeyError) as e:
        log_error(f"Could not load configuration from config.yaml. Error: {e}")
        return

    log_info(f"--- Inspecting Feature File: {features_path} ---")

    if not features_path.exists():
        log_error(
            f"File not found at {features_path}. Please ensure the 'build' command ran successfully."
        )
        return

    try:
        df = pd.read_csv(features_path)

        log_success(
            f"Successfully loaded file with {len(df)} rows and {len(df.columns)} columns."
        )

        print("\n" + "=" * 50)

        # 1. Verify the 'surface' column
        if "surface" in df.columns:
            log_info("Verifying 'surface' column distribution:")
            surface_counts = df["surface"].value_counts()
            print(surface_counts)
            # Check if there is more than one type of surface, which indicates success
            if len(surface_counts) > 1:
                log_success("✅ Verification PASSED: Multiple surface types found.")
            else:
                log_error(
                    "❌ Verification FAILED: Only one surface type found. The data may be incorrect."
                )
        else:
            log_error("❌ Verification FAILED: 'surface' column not found in the file.")

        print("=" * 50 + "\n")

        # 2. Display a sample of the data
        log_info("Displaying first 10 rows of the feature file:")
        print(df.head(10).to_string())

    except Exception as e:
        log_error(f"An error occurred while inspecting the file: {e}")


if __name__ == "__main__":
    inspect_feature_file()
