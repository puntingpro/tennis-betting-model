import pandas as pd
import yaml


def inspect_raw_players_file():
    """
    Loads the raw players CSV file defined in the config and prints its columns.
    """
    print("--- Inspecting Raw Players File Columns ---")
    try:
        # Load the config to find the path to the raw_players file
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)

        raw_players_path = config["data_paths"]["raw_players"]
        print(f"Found path to raw players file: {raw_players_path}")

    except (FileNotFoundError, KeyError):
        print("Error: Could not find config.yaml or the required path in it.")
        print("Please ensure your config.yaml is in the same directory and is valid.")
        return

    try:
        # Load a sample of the raw players file to see its structure
        players_df = pd.read_csv(raw_players_path, encoding="latin-1", nrows=5)

        print("\nSuccessfully loaded a sample of the raw players file.")
        print(f"The columns available in '{raw_players_path}' are:")
        print("-" * 30)
        for col in players_df.columns:
            print(col)
        print("-" * 30)

    except FileNotFoundError:
        print(
            f"Error: Could not find the file at the specified path: {raw_players_path}"
        )
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    inspect_raw_players_file()
