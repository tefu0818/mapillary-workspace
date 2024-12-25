import json
import logging
from pathlib import Path
from typing import Any, Dict
from abc import ABC, abstractmethod

import torch
from PIL import Image
from tqdm import tqdm
from transformers import BlipProcessor, BlipForConditionalGeneration
import cv2
from multiprocessing import Pool


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Annotator(ABC):
    @abstractmethod
    def annotate(self, input_json_path: Path, output_json_path: Path) -> None:
        pass


class BlipAnnotator(Annotator):
    def __init__(
        self,
        model: BlipForConditionalGeneration,
        processor: BlipProcessor,
        device: str = "cuda",
    ) -> None:
        self.processor = processor
        self.model = model.to(device)
        self.device = device

    @classmethod
    def from_model_path(
        cls,
        model_path: str = "Salesforce/blip-image-captioning-large",
        device: str = "cuda",
    ) -> "BlipAnnotator":
        logger.info(f"Loading processor from {model_path}")
        processor = BlipProcessor.from_pretrained(model_path)
        logger.info(f"Loading model from {model_path}")
        model = BlipForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16
        ).to(device)
        logger.info("Model and processor loaded successfully")
        return cls(model, processor, device)

    @torch.no_grad()
    def annotate(
        self,
        input_json_path: Path,
        output_json_path: Path,
        text_condition: str | None = "a photography of",
        batch_size: int = 8,
    ) -> None:
        with input_json_path.open("r") as f:
            data: Dict[str, Any] = json.load(f)

        features = data["features"]
        for i in tqdm(range(0, len(features), batch_size), desc="Captioning images"):
            batch_features = features[i : i + batch_size]
            images = []
            for feature in batch_features:
                properties = feature["properties"]
                image_path = input_json_path.parent / properties["image_path"]
                pil_img = Image.open(image_path).convert("RGB")
                images.append(pil_img)

            inputs = self.processor(images, return_tensors="pt", padding=True).to(
                self.device, torch.float16
            )
            out = self.model.generate(**inputs)
            captions = self.processor.batch_decode(out, skip_special_tokens=True)

            for feature, caption in zip(batch_features, captions):
                feature["properties"]["unconditional_blip_caption"] = caption

            if text_condition:
                inputs = self.processor(
                    images,
                    [text_condition] * len(images),
                    return_tensors="pt",
                    padding=True,
                ).to(self.device, torch.float16)
                out = self.model.generate(**inputs)
                conditional_captions = self.processor.batch_decode(
                    out, skip_special_tokens=True
                )

                for feature, conditional_caption in zip(
                    batch_features, conditional_captions
                ):
                    feature["properties"]["conditional_blip_caption"] = (
                        conditional_caption
                    )

        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with output_json_path.open("w") as f:
            json.dump(data, f, indent=4)


class SharpnessAnnotator(Annotator):
    def __init__(self):
        pass

    @staticmethod
    def calculate_sharpness(args: tuple) -> Dict[str, Any]:
        feature, input_json_path = args
        properties = feature["properties"]
        image_path = input_json_path.parent / properties["image_path"]

        image = cv2.imread(str(image_path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var() / (
            gray.shape[0] * gray.shape[1]
        )
        feature["properties"]["sharpness"] = sharpness
        return feature

    def annotate(
        self, input_json_path: Path, output_json_path: Path, num_workers: int = 4
    ) -> None:
        with input_json_path.open("r") as f:
            data: Dict[str, Any] = json.load(f)
        features = data["features"]

        with Pool(num_workers) as pool:
            features = list(
                tqdm(
                    pool.imap(
                        SharpnessAnnotator.calculate_sharpness,
                        [(feature, input_json_path) for feature in features],
                    ),
                    total=len(features),
                    desc="Calculating sharpness",
                )
            )

        data["features"] = features

        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        with output_json_path.open("w") as f:
            json.dump(data, f, indent=4)
