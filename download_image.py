import argparse
import time
import json
import logging
from pathlib import Path

import mapillary as mly
import requests
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


ORIGINAL_DATA_JSON_STR = "original_data.json"
DATA_JSON_STR = "data.json"
IMAGES_STR = "images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download images from Mapillary using the Mapillary Vector Tile API"
    )
    parser.add_argument("save_dir", type=Path, help="Directory to save images")
    parser.add_argument("access_token", type=str, help="Mapillary access token")
    parser.add_argument("west", type=float, help="Western longitude")
    parser.add_argument("south", type=float, help="Southern latitude")
    parser.add_argument("east", type=float, help="Eastern longitude")
    parser.add_argument("north", type=float, help="Northern latitude")
    return parser.parse_args()


def download_image(
    image_data: dict, save_dir: Path, resolution: int = 1024, chunk_size: int = 8192
) -> dict | None:
    try:
        image_id = image_data["properties"]["id"]
        image_url = mly.controller.image.get_image_thumbnail_controller(
            image_id=image_id, resolution=resolution
        )
        image_relative_path = f"{IMAGES_STR}/{image_id}.jpg"
        with requests.get(image_url, stream=True) as r:
            # r.raise_for_status()
            with open(save_dir / image_relative_path, "wb") as handler:
                for chunk in r.iter_content(chunk_size=chunk_size):
                    handler.write(chunk)

        with open(save_dir / image_relative_path, "rb") as handler:
            image = Image.open(handler)
        width, height = image.size
        image_data["properties"]["width"] = width
        image_data["properties"]["height"] = height
        image_data["properties"]["image_path"] = image_relative_path
        image_data["properties"]["url"] = image_url
        return image_data
    except Exception as e:
        logging.error(f"Error downloading image {image_id}: {e}")
        return None


def download_images(
    save_dir: Path,
    access_token: str,
    west: float,
    south: float,
    east: float,
    north: float,
    image_type: str = "flat",
    compass_angle: tuple = (0, 360),
    num_workers: int = 6,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=False)
    (save_dir / IMAGES_STR).mkdir()
    mly.interface.set_access_token(access_token)

    data = mly.interface.images_in_bbox(
        bbox={"west": west, "south": south, "east": east, "north": north},
        image_type=image_type,
        compass_angle=compass_angle,
    )
    data: dict = json.loads(data)

    # Save original data
    with open(save_dir / ORIGINAL_DATA_JSON_STR, "w") as f:
        json.dump(data, f, indent=4)

    # Download images
    successful_images = []

    def process_image(image_data):
        time.sleep(0.02)  # delay
        return download_image(image_data, save_dir)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(
            tqdm(
                executor.map(process_image, data["features"]),
                total=len(data["features"]),
                desc="Downloading images",
            )
        )

    for result in results:
        if result:
            successful_images.append(result)

    # Save data
    if successful_images:
        output_data = {
            "bbox": {"west": west, "south": south, "east": east, "north": north},
            "features": successful_images,
        }
        with open(save_dir / DATA_JSON_STR, "w") as f:
            json.dump(output_data, f, indent=4)


def main() -> None:
    """
    example usage: python download_images.py <save_dir> <access_token> <west> <south> <east> <north>
    """
    args = parse_args()
    download_images(
        args.save_dir, args.access_token, args.west, args.south, args.east, args.north
    )


if __name__ == "__main__":
    main()
