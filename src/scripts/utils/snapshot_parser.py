import bz2
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, TextIO


class SnapshotParser:
    """
    Parses Betfair snapshot files (plain text or .bz2) containing JSON lines.
    Modes:
      - "metadata": extracts marketDefinition metadata into a list of dicts
      - "ltp_only": extracts last-traded-price updates for each runner
      - "full": extracts all available runner-change fields
    """

    VALID_MODES = {"metadata", "ltp_only", "full"}

    def __init__(self, mode: str = "full"):
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Unknown mode: {mode!r}. Valid modes are: {self.VALID_MODES}"
            )
        self.mode = mode

    def parse_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Read the given file (plain text or .bz2) line by line, parse each JSON
        message, and return a list of row-dicts according to the selected mode.
        :param file_path: path to a .txt/.csv or .bz2 snapshot file
        :return: list of dicts containing the requested data for each mode
        """
        p = Path(file_path)
        opener: Callable[..., TextIO] = bz2.open if p.suffix == ".bz2" else open  # type: ignore[assignment]

        rows: List[Dict[str, Any]] = []
        with opener(file_path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    msg = json.loads(line)
                except json.JSONDecodeError:
                    continue  # skip malformed lines

                if msg.get("op") != "mcm":
                    continue

                for change in msg.get("mc", []):
                    market_id = change.get("id")
                    publish_time = msg.get("pt")
                    md = change.get("marketDefinition", {})

                    if self.mode == "metadata":
                        metadata_row: Dict[str, Any] = {
                            "market_id": market_id,
                            "market_time": md.get("marketTime"),
                            "market_name": md.get("name"),
                        }
                        for idx, runner in enumerate(md.get("runners", []), start=1):
                            metadata_row[f"runner_{idx}"] = runner.get("name")
                        rows.append(metadata_row)

                    else:  # self.mode is "ltp_only" or "full"
                        # Create a lookup map from selection ID to runner name for this market
                        id_to_name_map = {
                            runner.get("id"): runner.get("name")
                            for runner in md.get("runners", [])
                        }
                        
                        for rc in change.get("rc", []):
                            selection_id = rc.get("id")
                            runner_row: Dict[str, Any] = {
                                "market_id": market_id,
                                "timestamp": publish_time,
                                "selection_id": selection_id,
                                "runner_name": id_to_name_map.get(selection_id)
                            }

                            if "ltp" in rc:
                                runner_row["ltp"] = rc.get("ltp")

                            if self.mode == "full":
                                if "tv" in rc:
                                    runner_row["volume"] = rc["tv"]
                                if "atb" in rc:
                                    runner_row["best_available_to_back"] = rc["atb"]
                                if "atl" in rc:
                                    runner_row["best_available_to_lay"] = rc["atl"]
                            
                            rows.append(runner_row)

        return rows