# src/scripts/pipeline/run_automation.py

import time
import schedule
from datetime import datetime
from scripts.utils.logger import setup_logging, log_info
from scripts.utils.config import load_config
from scripts.pipeline.run_pipeline import run_pipeline_once


def job():
    """The main job to be scheduled."""
    log_info(
        f"--- Starting new pipeline run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---"
    )
    config = load_config("config.yaml")
    # Automation always runs in live mode (dry_run=False)
    # Alerts are handled by the process_markets function
    run_pipeline_once(config, dry_run=False)
    log_info("--- Pipeline run finished. Waiting for next schedule... ---")


def main(args):
    """Main CLI entrypoint for the automation service."""
    setup_logging()
    log_info("🚀 Starting PuntingPro Automation Service.")
    log_info("The pipeline will now run every 15 minutes.")

    job()
    schedule.every(15).minutes.do(job)

    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    # Create a dummy args object for standalone execution if needed
    class DummyArgs:
        pass

    args = DummyArgs()
    main(args)
