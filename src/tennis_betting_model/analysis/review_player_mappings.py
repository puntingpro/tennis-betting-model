import pandas as pd
import streamlit as st
from pathlib import Path
from thefuzz import process
from tennis_betting_model.utils.config import load_config
from tennis_betting_model.utils.logger import setup_logging
from tennis_betting_model.utils.data_loader import load_historical_player_data


# --- Main Streamlit Application ---
def run():
    st.set_page_config(layout="wide", page_title="Player Mapping Review")
    setup_logging()

    st.title(" Player Mapping Review Tool")
    st.markdown(
        "A tool to review and correct ambiguous player name mappings found by the fuzzy matching process."
    )

    try:
        config = load_config("config.yaml")
        paths = config["data_paths"]
        map_path = Path(paths["player_map"])
        raw_data_dir = Path(paths["raw_data_dir"])

        if not map_path.exists():
            st.error(
                f"Mapping file not found at '{map_path}'. Please run the 'create-player-map' command first."
            )
            return

        df_map = pd.read_csv(map_path)
        df_historical_players = load_historical_player_data(raw_data_dir)
        historical_player_list = df_historical_players["historical_name"].tolist()
        historical_player_lookup = df_historical_players.set_index("historical_name")[
            "historical_id"
        ].to_dict()

    except Exception as e:
        st.error(f"Failed to load necessary data. Error: {e}")
        return

    st.sidebar.header("Filter Options")
    confidence_threshold = st.sidebar.slider(
        "Show matches with confidence below:",
        min_value=80,
        max_value=100,
        value=98,
        step=1,
    )

    filter_method = st.sidebar.selectbox(
        "Filter by Match Method", options=["All", "Fuzzy", "Unique Lastname"], index=0
    )

    # Filter for ambiguous matches
    ambiguous_df = df_map[df_map["confidence"] < confidence_threshold]
    if filter_method != "All":
        ambiguous_df = ambiguous_df[ambiguous_df["method"] == filter_method]

    st.info(
        f"Found **{len(ambiguous_df)}** ambiguous mappings to review based on your filters."
    )

    if "corrections" not in st.session_state:
        st.session_state.corrections = {}

    for index, row in ambiguous_df.iterrows():
        st.divider()
        col1, col2, col3 = st.columns([2, 3, 1])

        betfair_name = row["betfair_name"]
        original_match = row["matched_name"]

        col1.markdown(f"**Betfair Name:**\n#### {betfair_name}")
        col1.markdown(
            f"**Current Match:** `{original_match}` ({row['confidence']:.1f}%)"
        )
        col1.markdown(f"**Method:** `{row['method']}`")

        # Find top 5 alternatives
        alternatives = [
            x[0] for x in process.extract(betfair_name, historical_player_list, limit=5)
        ]

        # Ensure the original match is in the list
        if original_match not in alternatives:
            alternatives.insert(0, original_match)

        # Use a unique key for each selectbox
        key = f"select_{row['betfair_id']}"

        # Set default index to the current match
        try:
            default_index = alternatives.index(original_match)
        except ValueError:
            default_index = 0

        corrected_name = col2.selectbox(
            label="Select the correct historical name:",
            options=alternatives,
            index=default_index,
            key=key,
        )

        if corrected_name != original_match:
            col3.warning("Changed")
            # Store the correction
            st.session_state.corrections[row["betfair_id"]] = corrected_name

    st.divider()

    if st.button("💾 Save All Changes", use_container_width=True, type="primary"):
        if not st.session_state.corrections:
            st.warning("No changes have been made.")
        else:
            # Apply corrections to the dataframe
            for betfair_id, new_name in st.session_state.corrections.items():
                new_hist_id = historical_player_lookup.get(new_name)
                df_map.loc[
                    df_map["betfair_id"] == betfair_id, "matched_name"
                ] = new_name
                df_map.loc[
                    df_map["betfair_id"] == betfair_id, "historical_id"
                ] = new_hist_id
                df_map.loc[df_map["betfair_id"] == betfair_id, "confidence"] = 100.0
                df_map.loc[
                    df_map["betfair_id"] == betfair_id, "method"
                ] = "Manual Correction"

            # Save the updated file
            df_map.to_csv(map_path, index=False)
            st.session_state.corrections = {}  # Clear corrections after saving
            st.success(
                f"✅ Successfully saved {len(st.session_state.corrections)} corrections to '{map_path}'!"
            )
            st.rerun()


if __name__ == "__main__":
    run()
