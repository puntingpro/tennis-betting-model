import time
import schedule
from datetime import datetime
from tennis_betting_model.utils.logger import setup_logging, log_info
from tennis_betting_model.utils.config import load_config
from tennis_betting_model.pipeline.run_pipeline import run_pipeline_once


def job():
    """The main job to be scheduled."""
    log_info(
        f"--- Starting new pipeline run at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ---"
    )
    config = load_config("config.yaml")
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

    class DummyArgs:
        pass

    args = DummyArgs()
    main(args)
