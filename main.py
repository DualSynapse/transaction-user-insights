"""Entry point for the whole pipeline: python main.py [--stage clean|features|segment|report] [--config config.yaml]"""
import argparse
import sys
import time
import traceback

from src.utils.config import load_config
from src.utils.tools import table_exists
from src.utils.logger import setup_logger

STAGES = ["clean", "features", "segment", "report"]


def parse_args():
    parser = argparse.ArgumentParser(description="Transaction User Insights pipeline")
    parser.add_argument("--stage", choices=STAGES, default=None,
                         help="Run a single stage instead of the whole pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    return parser.parse_args()


def run_clean(cfg, logger):
    from src.cleaning.pipeline import run_cleaning
    return run_cleaning(cfg)


def run_features(cfg, logger):
    if not table_exists(cfg.path("transactions_clean")):
        logger.error("Missing input: %s. Run the clean stage first (python main.py --stage clean).",
                     cfg.path("transactions_clean"))
        sys.exit(1)
    from src.features.pipeline import run_features as _run_features
    return _run_features(cfg)


def run_segment(cfg, logger):
    if not table_exists(cfg.path("user_features")):
        logger.error("Missing input: %s. Run the features stage first (python main.py --stage features).",
                     cfg.path("user_features"))
        sys.exit(1)
    from src.segmentation.pipeline import run_segmentation
    return run_segmentation(cfg)


def run_report(cfg, logger):
    if not table_exists(cfg.path("user_features")):
        logger.error("Missing input: %s. Run the features stage first (python main.py --stage features).",
                     cfg.path("user_features"))
        sys.exit(1)
    if not table_exists(cfg.path("user_segments")):
        logger.error("Missing input: %s. Run the segment stage first (python main.py --stage segment).",
                     cfg.path("user_segments"))
        sys.exit(1)
    from src.reporting.pipeline import run_report as _run_report
    return _run_report(cfg)


STAGE_FUNCS = {"clean": run_clean, "features": run_features, "segment": run_segment, "report": run_report}


def main():
    args = parse_args()
    cfg = load_config(args.config)
    logger = setup_logger(cfg.path("logs_dir"))

    stages = [args.stage] if args.stage else STAGES

    for stage in stages:
        logger.info("=== Starting stage: %s ===", stage)
        start = time.time()
        try:
            STAGE_FUNCS[stage](cfg, logger)
        except SystemExit:
            raise
        except Exception:
            logger.error("Stage '%s' failed:\n%s", stage, traceback.format_exc())
            sys.exit(1)
        elapsed = time.time() - start
        logger.info("=== Finished stage: %s (%.1fs) ===", stage, elapsed)


if __name__ == "__main__":
    main()
