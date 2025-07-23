import pandas as pd


def inspect_feature_file():
    """
    Loads the features CSV and prints its columns to see what's inside.
    """
    print("--- Inspecting Feature File Columns ---")
    try:
        # We read only the first 5 rows to speed things up, we only need the header
        features_df = pd.read_csv("data/processed/all_advanced_features.csv", nrows=5)

        print("\nSuccessfully loaded a sample of the features file.")
        print("The columns available in 'all_advanced_features.csv' are:")
        print("-" * 30)
        for col in features_df.columns:
            print(col)
        print("-" * 30)

    except FileNotFoundError:
        print("Error: Could not find data/processed/all_advanced_features.csv")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    inspect_feature_file()
