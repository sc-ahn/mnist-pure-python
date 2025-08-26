import numpy as np
from PIL import Image

from src.model import MNIST_CNN
from src.settings import CKPT_PATH


def main(data: np.ndarray):
    model = MNIST_CNN.load_weights(CKPT_PATH)
    pred = model.predict_with_ndarray(data)
    print(pred)


if __name__ == "__main__":
    import pathlib
    import sys

    if len(sys.argv) != 2:
        print("Usage: python inference.py <image_path>")
        sys.exit(1)

    image_path = pathlib.Path(sys.argv[1])
    image = Image.open(image_path).convert("L")  # Grayscale로 변환
    image = image.resize((28, 28))  # 모델 입력 크기에 맞게 조정
    image_np = np.array(image).reshape(1, 1, 28, 28) / 255.0
    main(image_np)
