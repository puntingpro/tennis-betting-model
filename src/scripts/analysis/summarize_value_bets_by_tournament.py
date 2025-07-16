# src/scripts/analysis/summarize_value_bets_by_tournament.py

from pathlib import Path
import pandas as pd
import argparse
from typing import Dict, Optional

from src.scripts.utils.file_utils import load_dataframes
from src.scripts.utils.logger import log_error, log_success, setup_logging
from src.scripts.utils.config import load_config

def get_tournament_category(tourney_name: str) -> str:
    """
    Categorizes a tournament name into a broader category for better analysis.
    """
    tourney_name = str(tourney_name).lower()
    
    category_map: Dict[str, str] = {
        'grand slam': 'Grand Slam', 'australian open': 'Grand Slam', 'roland garros': 'Grand Slam',
        'french open': 'Grand Slam', 'wimbledon': 'Grand Slam', 'us open': 'Grand Slam',
        'masters': 'Masters 1000', 'tour finals': 'Tour Finals', 'next gen finals': 'Tour Finals',
        'atp cup': 'Team Event', 'davis cup': 'Team Event', 'laver cup': 'Team Event',
        'olympics': 'Olympics', 'challenger': 'Challenger', 'itf': 'ITF / Futures'
    }

    for keyword, category in category_map.items():
        if keyword in tourney_name:
            return category
            
    return 'ATP / WTA Tour'


def run_summarize_by_tournament(df: pd.DataFrame, min_bets: int = 1) -> pd.DataFrame:
    """
    Summarizes backtest results by tournament category, calculating profit and ROI.
    """
    if df.empty:
        return pd.DataFrame()

    df['tourney_category'] = df['tourney_name'].apply(get_tournament_category)
    
    # --- BUG FIX ---
    # The 'winner' column already indicates if the bet was successful (1) or not (0).
    # Profit calculation should be based directly on this outcome.
    df['profit'] = df.apply(
        lambda row: (row['odds'] - 1) if row['winner'] == 1 else -1,
        axis=1
    )
    # --- END FIX ---
    
    df['stake'] = 1

    tournament_summary = (
        df.groupby("tourney_category")
        .agg(
            total_bets=("stake", "sum"),
            total_profit=("profit", "sum"),
            tournaments=("tourney_name", lambda x: sorted(x.unique().tolist()))
        )
        .reset_index()
    )
    tournament_summary["roi"] = (
        (tournament_summary["total_profit"] / tournament_summary["total_bets"]) * 100
    )
    
    filtered_summary = tournament_summary[tournament_summary['total_bets'] >= min_bets]
    
    return filtered_summary.sort_values(by="roi", ascending=False)

def main_cli(args: argparse.Namespace) -> None:
    """
    Main CLI entrypoint for summarizing backtest results by tournament category.
    """
    setup_logging()
    config = load_config(args.config)
    paths = config['data_paths']

    df = load_dataframes(paths['backtest_results'])
    summary_df = run_summarize_by_tournament(df, args.min_bets)

    if not summary_df.empty:
        log_success("✅ Successfully summarized tournaments by category.")
        
        display_df = summary_df.copy()
        if not args.show_tournaments:
            display_df = display_df.drop(columns=['tournaments'])

        print("\n--- Tournament Category Performance (min_bets={}) ---".format(args.min_bets))
        print(display_df.to_string())
        
        if args.show_tournaments:
            print("\n--- Detailed Tournament Lists ---")
            for _, row in summary_df.iterrows():
                print(f"\nCategory: {row['tourney_category']}")
                for name in row['tournaments']:
                    print(f"  - {name}")
            print("="*80)

        output_path = Path(paths['tournament_summary'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_path, index=False)
        log_success(f"✅ Saved summary data to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize backtest results by tournament category to calculate profit and ROI."
    )
    parser.add_argument("--config", default="config.yaml", help="Path to the config file.")
    parser.add_argument(
        "--min-bets", type=int, default=100,
        help="The minimum number of total bets required to include a tournament category in the summary."
    )
    parser.add_argument(
        "--show-tournaments", action="store_true",
        help="If set, shows the full list of individual tournaments within each category."
    )
    args = parser.parse_args()
    main_cli(args)