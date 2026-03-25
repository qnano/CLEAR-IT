"""
Run the public embedding-generation pipeline from a YAML recipe.

Usage:
    python clearit/scripts/run_embed_pipeline.py --recipe path/to/01_embed.yaml
"""

from pathlib import Path
import argparse

from clearit.embeddings import run_embed_recipe


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CLEAR-IT embedding HDF5 files from a recipe")
    parser.add_argument("--recipe", required=True, type=str, help="Path to the embed recipe YAML")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing HDF5 outputs instead of skipping them",
    )
    args = parser.parse_args()

    run_embed_recipe(Path(args.recipe), overwrite=args.overwrite)


if __name__ == "__main__":
    main()
