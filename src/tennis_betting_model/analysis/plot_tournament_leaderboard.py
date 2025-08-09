# src/tennis_betting_model/analysis/plot_tournament_leaderboard.py

from pathlib import Path
import pandas as pd
import plotly.express as px
from src.tennis_betting_model.utils.logger import log_error, log_success, setup_logging
from src.tennis_betting_model.utils.config_schema import Config


def run_plot_leaderboard(df: pd.DataFrame, sort_by: str, top_n: int):
    """
    Generates an interactive bar chart of tournament categories using Plotly.
    """
    df_sorted = df.sort_values(by=sort_by, ascending=True).tail(top_n)

    fig = px.bar(
        df_sorted,
        x=sort_by,
        y="tourney_category",
        orientation="h",
        title=f"Top {top_n} Tournament Categories by {sort_by.title()}",
        labels={
            "tourney_category": "Tournament Category",
            sort_by: sort_by.replace("_", " ").title(),
        },
        text=df_sorted[sort_by].apply(
            lambda x: f"{x:.2f}%" if sort_by == "roi" else f"{x:.2f}"
        ),
    )
    fig.update_layout(
        yaxis_title="Tournament Category",
        xaxis_title=sort_by.replace("_", " ").title(),
        title_font_size=20,
        height=800,
    )
    return fig


def main_cli(config: Config, show_plot: bool):
    """
    Main function for plotting the leaderboard, driven by the config file.
    """
    setup_logging()
    paths = config.data_paths
    analysis_params = config.analysis_params
    leaderboard_top_n = analysis_params.leaderboard_top_n

    try:
        input_path = Path(paths.tournament_summary)
        df = pd.read_csv(input_path)

        fig = run_plot_leaderboard(df, sort_by="roi", top_n=leaderboard_top_n)

        plot_dir = Path(paths.plot_dir)
        output_path = (
            plot_dir / "tournament_leaderboard.html"
        )  # Save as HTML for interactivity
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig.write_html(str(output_path))
        log_success(f"Saved interactive leaderboard plot to {output_path}")

        if show_plot:
            fig.show()

    except FileNotFoundError:
        log_error(f"Input file not found: {paths.tournament_summary}")
    except Exception as e:
        log_error(f"An unexpected error occurred: {e}")
