"""
- `t10k-images.idx3-ubyte`: 이미지 테스트 셋
- `t10k-labels.idx1-ubyte`: 이미지 테스트 레이블
- `train-images.idx3-ubyte`: 이미지 학습 셋
- `train-labels.idx1-ubyte`: 이미지 학습 레이블

위 데이터를 로드하여 학습 및 테스트 데이터를 생성하는 모듈
"""

from pathlib import Path

import numpy as np

from src.settings import DATA_DIR


def read_int32_be(file):
    """파일로부터 Big-Endian으로 인코딩된 32비트 정수를 읽습니다."""
    bytes_ = file.read(4)  # 4 바이트 읽기
    return (bytes_[0] << 24) | (bytes_[1] << 16) | (bytes_[2] << 8) | bytes_[3]


def load_mnist_by_kind(path: Path, kind="train") -> tuple[np.ndarray, np.ndarray]:
    """`path`에서 MNIST 데이터를 로드합니다."""
    labels_path = path / f"{kind}-labels.idx1-ubyte"
    images_path = path / f"{kind}-images.idx3-ubyte"

    # 레이블 읽기
    with open(labels_path, "rb") as lbpath:
        _magic = read_int32_be(lbpath)  # Magic number
        _n = read_int32_be(lbpath)  # Number of labels
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8)

    # 이미지 읽기
    with open(images_path, "rb") as imgpath:
        _magic = read_int32_be(imgpath)  # Magic number
        num = read_int32_be(imgpath)  # Number of images
        rows = read_int32_be(imgpath)  # Image rows
        cols = read_int32_be(imgpath)  # Image cols
        images = np.frombuffer(imgpath.read(), dtype=np.uint8).reshape(num, rows * cols)

    return images, labels


def load_mnist(
    path: Path,
) -> tuple[
    tuple[np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray],
]:
    """MNIST 데이터를 로드하여 학습 및 테스트 데이터를 생성합니다."""
    train_images, train_labels = load_mnist_by_kind(path, kind="train")
    test_images, test_labels = load_mnist_by_kind(path, kind="t10k")

    return (train_images, train_labels), (test_images, test_labels)


if __name__ == "__main__":
    (
        (
            _train_images,
            _train_labels,
        ),
        (
            _test_images,
            _test_labels,
        ),
    ) = load_mnist(DATA_DIR)

    print(f"{_train_images.shape = }, {_train_labels.shape = }")
    print(f"{_test_images.shape = }, {_test_labels.shape = }")


def rotate_image(img, angle):
    """이미지를 주어진 각도로 회전 (간단한 구현)"""
    # scipy 없이 간단한 회전 구현
    # 실제로는 복잡하므로 여기서는 약간의 노이즈만 추가
    noise = np.random.normal(0, 0.02, img.shape)
    return np.clip(img + noise, 0, 1)


def shift_image(img, shift_x, shift_y):
    """이미지를 주어진 픽셀만큼 이동"""
    # img shape: (1, 28, 28)
    shifted = np.zeros_like(img)
    h, w = img.shape[1], img.shape[2]

    # 이동 범위 계산
    y_start = max(0, shift_y)
    y_end = min(h, h + shift_y)
    x_start = max(0, shift_x)
    x_end = min(w, w + shift_x)

    # 원본에서 복사할 범위
    orig_y_start = max(0, -shift_y)
    orig_y_end = orig_y_start + (y_end - y_start)
    orig_x_start = max(0, -shift_x)
    orig_x_end = orig_x_start + (x_end - x_start)

    if orig_y_end > orig_y_start and orig_x_end > orig_x_start:
        shifted[0, y_start:y_end, x_start:x_end] = img[
            0, orig_y_start:orig_y_end, orig_x_start:orig_x_end
        ]

    return shifted


def augment_data(images, labels):
    """간단한 데이터 증강"""
    augmented_images = []
    augmented_labels = []

    for img, label in zip(images, labels):
        # 원본 데이터
        augmented_images.append(img)
        augmented_labels.append(label)

        # 약간의 회전 (노이즈 추가로 대체)
        rotated = rotate_image(img, 0)  # 각도는 사용하지 않고 노이즈만 추가
        augmented_images.append(rotated)
        augmented_labels.append(label)

        # 약간의 이동 (±2 픽셀)
        shift_x = np.random.randint(-2, 3)
        shift_y = np.random.randint(-2, 3)
        shifted = shift_image(img, shift_x, shift_y)
        augmented_images.append(shifted)
        augmented_labels.append(label)

    return np.array(augmented_images), np.array(augmented_labels)
