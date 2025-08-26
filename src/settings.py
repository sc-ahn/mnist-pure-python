from typing import Annotated
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CKPT_DIR = BASE_DIR / "ckpt"
CKPT_NAME = 'ckpt.pkl'
CKPT_PATH = CKPT_DIR / CKPT_NAME
DATA_DIR = BASE_DIR / "dataset"
IMG_DIR = BASE_DIR / "images"

__all__ = ["CKPT_DIR", "DATA_DIR", "IMG_DIR", "BASE_DIR", "CKPT_PATH"]


class Params:
    BATCH_SIZE: Annotated[int, "배치 사이즈"] = 128
    EPOCHS: Annotated[int, "학습 횟수"] = 20
    LEARNING_RATE: Annotated[float, "학습률"] = 0.005
    SUBSET_SIZE: Annotated[int | None, "훈련 하위 집합 크기"] = None  # For training subset, None for full dataset
