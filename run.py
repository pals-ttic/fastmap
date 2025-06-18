import argparse

from fastmap.config import Config, load_config
from fastmap.engine import engine


def run(args):
    if args.config is None:
        config = Config()
    else:
        config: Config = load_config(args.config)

    engine(
        cfg=config,
        device=args.device,
        database_path=args.database,
        output_dir=args.output_dir,
        pinhole=args.pinhole,
        headless=args.headless,
        calibrated=args.calibrated,
        image_dir=args.image_dir,
        gt_model_path=args.gt_model,
    )


if __name__ == "__main__":
    # parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--database", type=str, required=True, help="Path to input database."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file. See fastmap/config.py for all the options.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory containing images.",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Path to output directory."
    )
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use.")
    parser.add_argument(
        "--pinhole", action="store_true", help="Use pinhole camera model."
    )
    parser.add_argument("--headless", action="store_true", help="Run in headless mode.")
    parser.add_argument(
        "--calibrated",
        action="store_true",
        help="Use the focal length and principal point from the database.",
    )
    parser.add_argument(
        "--gt_model",
        type=str,
        default=None,
        help="Path to ground truth model (e.g. otuput of colmap) for debugging purposes.",
    )
    args = parser.parse_args()

    # run the engine
    run(args)
