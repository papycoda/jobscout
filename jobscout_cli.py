#!/usr/bin/env python
"""
JobScout CLI entry point.

Usage:
    python jobscout_cli.py              # Run once
    python jobscout_cli.py --schedule   # Start scheduler
    python jobscout_cli.py --config my-config.yaml
"""

import sys
import argparse
from jobscout.main import JobScout
from jobscout.config import load_config
from jobscout.scheduler import JobScoutScheduler


def main():
    parser = argparse.ArgumentParser(
        description="JobScout: A conservative job-search assistant"
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to config file (default: config.yaml)"
    )
    parser.add_argument(
        "--schedule",
        action="store_true",
        help="Run in scheduled mode (daemon)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once immediately (default behavior)"
    )

    args = parser.parse_args()

    try:
        # Load configuration
        print(f"Loading configuration from {args.config}...")
        config = load_config(args.config)

        # Create JobScout instance
        jobscout = JobScout(config)

        if args.schedule:
            # Run in scheduled mode
            print("Starting JobScout scheduler...")
            print("Press Ctrl+C to stop")

            scheduler = JobScoutScheduler(config, jobscout.run_search)
            scheduler.start()
        else:
            # Run once
            print("Running JobScout job search...")
            success = jobscout.run_search()

            if success:
                print("\nJobScout completed successfully!")
                sys.exit(0)
            else:
                print("\nJobScout encountered errors. Check logs for details.")
                sys.exit(1)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nMake sure your config file exists. You can:")
        print("  1. Copy config.example.yaml to config.yaml")
        print("  2. Edit config.yaml with your details")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
