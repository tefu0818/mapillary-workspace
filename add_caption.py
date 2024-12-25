from pathlib import Path
import argparse

from src.annotator import BlipAnnotator


def process_json(
    input_json_file: Path, output_json_file: Path, model_path: str, device: str = "cuda"
) -> None:
    annotator = BlipAnnotator.from_model_path(model_path, device)
    annotator.annotate(input_json_file, output_json_file)


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Process JSON files.")
    parser.add_argument("input_json_file", type=Path, help="Input JSON file path")
    parser.add_argument("output_json_file", type=Path, help="Output JSON file path")
    parser.add_argument(
        "--model_path",
        type=str,
        default="Salesforce/blip-image-captioning-large",
        help="Path to the BLIP model",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run the model on"
    )
    return parser


def main() -> None:
    parser = create_parser()
    args = parser.parse_args()
    process_json(args.input_json_file, args.output_json_file, args.model_path, args.device)


if __name__ == "__main__":
    main()
