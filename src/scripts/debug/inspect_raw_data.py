# src/scripts/debug/inspect_raw_data.py

import pandas as pd
import argparse
import glob
import yaml
import sys
from pathlib import Path

# --- Add project root to the Python path to allow importing from src ---
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

def inspect_raw_match(match_id: str, raw_data_glob: str):
    """
    Finds and displays the raw data for a specific match_id from the source CSV files.
    """
    try:
        year = match_id.split('-')[0]
        tourney_id = '-'.join(match_id.split('-')[:2])
        match_num = int(match_id.split('-')[2])
    except (IndexError, ValueError):
        print(f"Error: Invalid match_id format. Expected 'YYYY-TourneyID-MatchNum', but got '{match_id}'.")
        return

    file_pattern = raw_data_glob.replace('*', year)
    matching_files = glob.glob(file_pattern)

    if not matching_files:
        print(f"Error: No raw data file found for the year {year} using pattern '{file_pattern}'.")
        return

    raw_file_path = matching_files[0]
    print(f"--- Inspecting Raw Data for match_id: {match_id} ---")
    print(f"Loading data from: {raw_file_path}\n")

    try:
        df_raw = pd.read_csv(raw_file_path, encoding='latin-1', low_memory=False)
    except FileNotFoundError:
        print(f"Error: File not found at '{raw_file_path}'")
        return

    match_data = df_raw[
        (df_raw['tourney_id'] == tourney_id) &
        (df_raw['match_num'] == match_num)
    ]

    if match_data.empty:
        print(f"Result: No match found for tourney_id '{tourney_id}' and match_num '{match_num}'.")
    else:
        columns_to_show = [
            'tourney_id', 'match_num', 'winner_name', 'loser_name',
            'winner_rank', 'winner_rank_points',
            'loser_rank', 'loser_rank_points'
        ]
        columns_to_show = [col for col in columns_to_show if col in df_raw.columns]
        print("Found Match Data:")
        print(match_data[columns_to_show].to_string(index=False))


def list_tournaments(backtest_file: str, output_path: str):
    """
    Lists all unique tournament names from the backtest results to a file.
    """
    try:
        df = pd.read_csv(backtest_file)
        unique_tournaments = sorted(df['tourney_name'].unique())
        
        # Write the list to the specified output file
        with open(output_path, 'w') as f:
            for name in unique_tournaments:
                f.write(f"{name}\n")
        
        print(f"✅ Successfully wrote {len(unique_tournaments)} unique tournament names to '{output_path}'.")

    except FileNotFoundError:
        print(f"Error: '{backtest_file}' not found. Please run a backtest first.")

def main():
    """
    Main function to parse arguments and run the inspection.
    """
    parser = argparse.ArgumentParser(
        description="Inspect raw data or list tournaments from backtest results.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    # Use subparsers for different actions
    subparsers = parser.add_subparsers(dest="action", required=True, help="Available actions")

    # Create the parser for the "list" command
    parser_list = subparsers.add_parser('list', help='List all unique tournaments to a file.')
    parser_list.add_argument("--output", default="tournaments.txt", help="Path to save the output text file (default: tournaments.txt).")
    parser_list.add_argument("--config", default="config.yaml", help="Path to the project's config file.")

    # Create the parser for the "inspect" command
    parser_inspect = subparsers.add_parser('inspect', help='Inspect a specific match in the raw data files.')
    parser_inspect.add_argument("match_id", type=str, help="The match_id to inspect (e.g., '2001-D011-1').")
    parser_inspect.add_argument("--config", default="config.yaml", help="Path to the project's config file.")
    
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: Could not load config file '{args.config}'.")
        return

    if args.action == 'list':
        backtest_file = config['data_paths']['backtest_results']
        list_tournaments(backtest_file, args.output)
    elif args.action == 'inspect':
        raw_matches_glob = config['data_paths']['raw_matches_glob']
        inspect_raw_match(args.match_id, raw_matches_glob)

if __name__ == "__main__":
    main()