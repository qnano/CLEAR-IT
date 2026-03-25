"""Run the CLEAR-IT MAPS benchmark stage from a YAML recipe.

Usage:
    python clearit/scripts/run_maps_benchmark.py --recipe path/to/01_benchmark.yaml
"""

from pathlib import Path
import argparse

from clearit.maps_benchmark import run_maps_benchmark_recipe


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the MAPS benchmark from a CLEAR-IT recipe")
    parser.add_argument("--recipe", required=True, type=str, help="Path to the benchmark recipe YAML")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing benchmark outputs instead of skipping them",
    )
    args = parser.parse_args()
    run_maps_benchmark_recipe(Path(args.recipe), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
