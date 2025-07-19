import pandas as pd
import matplotlib.pyplot as plt


def explore_tennis_data(csv_path="tennis_data.csv"):
    """
    Loads the tennis data and performs basic exploratory data analysis.
    """
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: The file {csv_path} was not found.")
        return

    # --- Initial Data Inspection ---
    print("\n--- Data Overview ---")
    print("First 5 rows of the dataset:")
    print(df.head())

    print("\nDataset Information (columns, data types, non-null counts):")
    df.info()

    # --- Analyzing a Single Market ---
    print("\n--- Single Market Analysis ---")
    # Find a market with a decent number of updates to analyze
    if df.empty:
        print("The dataset is empty. Cannot perform analysis.")
        return

    market_id_to_analyze = df["market_id"].value_counts().idxmax()
    market_df = df[df["market_id"] == market_id_to_analyze].copy()

    # Convert timestamp to datetime for plotting
    market_df["pt"] = pd.to_datetime(market_df["pt"])
    market_df.sort_values("pt", inplace=True)

    # Get runner names for the legend
    runner_names = market_df[["runner_id", "runner_name"]].drop_duplicates()
    print(f"Analyzing Market ID: {market_id_to_analyze}")
    print("Runners in this market:")
    print(runner_names)

    # --- Plotting the Odds Movement ---
    print("\nPlotting odds movement over time for the selected market...")

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, ax = plt.subplots(figsize=(15, 8))

    # --- FIX: Plot 'best_back_price' instead of 'ltp' ---
    for runner_id in runner_names["runner_id"]:
        runner_df = market_df[market_df["runner_id"] == runner_id]
        runner_name = runner_names[runner_names["runner_id"] == runner_id][
            "runner_name"
        ].iloc[0]
        ax.plot(
            runner_df["pt"],
            runner_df["best_back_price"],
            marker="o",
            linestyle="-",
            markersize=4,
            label=f"{runner_name}",
        )

    ax.set_title(f"Odds Movement for Market: {market_id_to_analyze}", fontsize=16)
    ax.set_xlabel("Time (UTC)", fontsize=12)
    ax.set_ylabel("Best Available Back Price", fontsize=12)  # FIX: Updated Y-axis label
    ax.legend()
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the plot to a file
    plot_filename = f"market_{market_id_to_analyze}_odds.png"
    plt.savefig(plot_filename)
    print(f"Plot saved as '{plot_filename}'")

    plt.show()


if __name__ == "__main__":
    explore_tennis_data()
