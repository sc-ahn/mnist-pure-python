from src.data import load_mnist_by_kind
from src.model import MNIST_CNN
from src.settings import CKPT_PATH, DATA_DIR

# 테스트 데이터 로드
test_images, test_labels = load_mnist_by_kind(DATA_DIR, "t10k")

# 이미지 정규화 및 reshape
test_images = test_images.reshape(-1, 1, 28, 28) / 255.0

# 모델 로드
model = MNIST_CNN.load_weights(CKPT_PATH)

# 모델 평가
model.evaluate(test_images, test_labels)
