from pathlib import Path
import argparse

from src.annotator import SharpnessAnnotator


def process_json(input_json_file: Path, output_json_file: Path) -> None:
    annotator = SharpnessAnnotator()
    annotator.annotate(input_json_file, output_json_file)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process JSON files.")
    parser.add_argument("input_json_file", type=Path, help="Input JSON file path")
    parser.add_argument("output_json_file", type=Path, help="Output JSON file path")
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    process_json(args.input_json_file, args.output_json_file)


if __name__ == "__main__":
    main()
