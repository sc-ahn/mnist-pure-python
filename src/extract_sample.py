"""
- `t10k-images.idx3-ubyte`: 이미지 테스트 셋
- `t10k-labels.idx1-ubyte`: 이미지 테스트 레이블

테스트 데이터 일부를 추출하여 저장하는 모듈
"""

import random
from PIL import Image

from src.settings import DATA_DIR, IMG_DIR
from src.data import load_mnist_by_kind


def extract_sample_images(num_samples: int = 10):
    """
    테스트 데이터셋에서 임의의 샘플을 추출하여 IMG_DIR에 저장합니다.

    Args:
        num_samples (int): 추출할 샘플 개수 (기본값: 10)
    """
    # IMG_DIR이 존재하지 않으면 생성
    IMG_DIR.mkdir(parents=True, exist_ok=True)

    # 테스트 데이터 로드
    test_images, test_labels = load_mnist_by_kind(DATA_DIR, kind="t10k")

    # 전체 테스트 데이터에서 임의의 인덱스 선택
    total_samples = len(test_images)
    if num_samples > total_samples:
        print(
            f"경고: 요청한 샘플 수({num_samples})가 전체 테스트 데이터({total_samples})보다 큽니다."
        )
        num_samples = total_samples

    # 중복 없이 임의의 인덱스 선택
    selected_indices = random.sample(range(total_samples), num_samples)

    print(f"테스트 데이터에서 {num_samples}개 샘플을 추출합니다...")

    # 선택된 샘플들을 이미지로 저장
    for i, idx in enumerate(selected_indices):
        # 28x28 이미지로 reshape (MNIST 이미지는 784차원 벡터로 저장됨)
        image_data = test_images[idx].reshape(28, 28)
        label = test_labels[idx]

        # PIL Image로 변환 (0-255 범위의 grayscale)
        image = Image.fromarray(image_data)

        # 파일명: sample_{순번}_{레이블}.png
        filename = f"sample_{i+1:03d}_label_{label}.png"
        filepath = IMG_DIR / filename

        # 이미지 저장
        image.save(filepath)
        print(f"저장됨: {filename} (레이블: {label})")

    print(f"\n모든 샘플이 {IMG_DIR}에 저장되었습니다.")


if __name__ == "__main__":
    # 기본값으로 10개 샘플 추출
    extract_sample_images()

    # 또는 원하는 개수 지정
    # extract_sample_images(num_samples=5)
