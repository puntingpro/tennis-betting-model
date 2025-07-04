# src/scripts/pipeline/run_full_pipeline.py

import traceback
from pathlib import Path
import concurrent.futures

import joblib
import pandas as pd

from scripts.pipeline.stages import STAGE_FUNCS
from scripts.utils.config import load_config
from scripts.utils.constants import (
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_EV_THRESHOLD,
    DEFAULT_INITIAL_BANKROLL,
    DEFAULT_MAX_MARGIN,
    DEFAULT_MAX_ODDS,
)
from scripts.utils.logger import log_error, log_info, log_success, log_warning


def resolve_stage_paths(label_cfg, working_dir):
    """
    Returns a dict mapping expected file types to resolved paths for a label.
    It dynamically discovers all needed keys from the STAGE_FUNCS definition.
    """
    paths = {}
    label = label_cfg["label"]

    all_input_keys = {
        key for stage in STAGE_FUNCS.values() for key in stage["input_keys"]
    }
    all_output_keys = {stage["output_key"] for stage in STAGE_FUNCS.values()}
    all_keys = list(all_input_keys.union(all_output_keys))

    all_keys.append("model_file")

    for key in all_keys:
        if key in label_cfg:
            paths[key] = Path(label_cfg[key])
        elif key.endswith("_csv"):
            base = working_dir / f"{label}_{key.replace('_csv', '')}.csv"
            paths[key] = base

    return paths


def run_pipeline_for_label(
    label_cfg,
    stages,
    working_dir,
    dry_run=False,
    overwrite=False,
    verbose=False,
    json_logs=False,
    only=None,
):
    label = label_cfg["label"]
    log_info(f"\n🏷️  Starting pipeline for label: {label}")
    resolved_paths = resolve_stage_paths(label_cfg, working_dir)
    stage_output_paths = {}
    pipeline_ok = True

    for stage in stages:
        if only and stage not in only:
            continue

        stage_info = STAGE_FUNCS.get(stage)
        if not stage_info:
            log_warning(f"⚠️  Unknown stage '{stage}', skipping.")
            continue

        log_info(f"--- Stage: {stage} ---")
        fn = stage_info["fn"]
        input_keys = stage_info["input_keys"]
        output_key = stage_info["output_key"]
        output_path = resolved_paths.get(output_key)

        if not output_path and output_key:
            log_error(
                f"❌ Could not resolve output path for key '{output_key}' in stage '{stage}'."
            )
            pipeline_ok = False
            break

        if not overwrite and not dry_run and output_path and output_path.exists():
            log_info(f"Skipping stage '{stage}': output file already exists.")
            stage_output_paths[output_key] = output_path
            continue

        try:
            input_paths = {}
            for k in input_keys:
                input_paths[k] = resolved_paths.get(k) or stage_output_paths.get(k)
                if not input_paths[k]:
                    raise RuntimeError(f"Missing input '{k}' for stage '{stage}'")

            log_info(f"Running {stage} with inputs: {input_paths}")

            result = None
            if dry_run:
                log_info(f"[DRY-RUN] Would write to {output_path}")
            # --- UPDATED SECTION: ADDED HANDLER FOR 'features' STAGE ---
            elif stage == "features":
                df_merged = pd.read_csv(input_paths["merged_matches_csv"])
                df_player_features = pd.read_csv(input_paths["player_features_csv"])
                result = fn(df_merged, df_player_features)
            # --- END UPDATED SECTION ---
            elif stage == "detect":
                df_pred = pd.read_csv(input_paths["predictions_csv"])
                params = {
                    "ev_threshold": label_cfg.get("ev_threshold", DEFAULT_EV_THRESHOLD),
                    "confidence_threshold": label_cfg.get(
                        "confidence_threshold", DEFAULT_CONFIDENCE_THRESHOLD
                    ),
                    "max_odds": label_cfg.get("max_odds", DEFAULT_MAX_ODDS),
                    "max_margin": label_cfg.get("max_margin", DEFAULT_MAX_MARGIN),
                }
                result = fn(df_pred, **params)
            elif stage == "simulate":
                df_bets = pd.read_csv(input_paths["value_bets_csv"])
                params = {
                    "initial_bankroll": label_cfg.get(
                        "initial_bankroll", DEFAULT_INITIAL_BANKROLL
                    ),
                    "strategy": label_cfg.get("staking_strategy", "kelly"),
                    "flat_stake_unit": label_cfg.get("flat_stake_unit", 10.0),
                }
                result = fn(df_bets, **params)
            else:
                func_args = []
                for key in input_keys:
                    if key == "model_file":
                        model_path = input_paths.get(key)
                        if not model_path or not Path(model_path).exists():
                           raise FileNotFoundError(f"Model file not found at path: {model_path}")
                        func_args.append(joblib.load(model_path))
                    elif input_paths.get(key):
                        func_args.append(pd.read_csv(input_paths[key]))
                result = fn(*func_args)

            if result is not None and output_path:
                output_path.parent.mkdir(parents=True, exist_ok=True)
                result.to_csv(output_path, index=False)
                log_success(f"✅ Stage '{stage}' complete. Output: {output_path}")

            if output_key:
                stage_output_paths[output_key] = output_path

        except Exception as e:
            log_error(f"❌ Stage '{stage}' failed for {label}: {e}")
            if verbose:
                traceback.print_exc()
            pipeline_ok = False
            break

    if pipeline_ok:
        log_success(f"🎉 Pipeline finished successfully for label: {label}")
    else:
        log_error(f"💔 Pipeline failed for label: {label}")
    
    return pipeline_ok


def main(config, only, batch, dry_run, overwrite, verbose, json_logs, working_dir):
    """Main orchestrator for running the pipeline."""
    app_cfg = load_config(config)
    stages = app_cfg.get("stages", ["build", "ids", "player_features", "merge", "features", "predict", "detect", "simulate"])
    working_dir = Path(working_dir)
    working_dir.mkdir(parents=True, exist_ok=True)

    defaults = app_cfg.get("defaults", {})
    tournaments = app_cfg.get("tournaments", [])

    if batch:
        labels_to_run = [{**defaults, **label_cfg} for label_cfg in tournaments]
    else:
        single_run_cfg = {**defaults, **app_cfg.get("pipeline", {})}
        labels_to_run = [single_run_cfg]

    log_info(
        f"Pipeline configured to run for {len(labels_to_run)} label(s). Stages: {stages}"
    )
    if only:
        log_warning(f"Running only a subset of stages: {only}")

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                run_pipeline_for_label,
                label_cfg,
                stages,
                working_dir,
                dry_run,
                overwrite,
                verbose,
                json_logs,
                only=only,
            ): label_cfg.get("label", "Unknown") for label_cfg in labels_to_run
        }

        for future in concurrent.futures.as_completed(futures):
            label = futures[future]
            try:
                result_ok = future.result()
                if not result_ok:
                    log_error(f"Sub-pipeline for {label} completed with errors.")
            except Exception as exc:
                log_error(f"{label} generated an exception: {exc}")

    log_info("All pipeline runs complete.")