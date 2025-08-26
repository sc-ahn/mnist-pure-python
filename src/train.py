import numpy as np

from src.data import augment_data, load_mnist_by_kind
from src.model import MNIST_CNN
from src.settings import CKPT_PATH, DATA_DIR, Params

# 학습용 데이터 로드
train_images, train_labels = load_mnist_by_kind(DATA_DIR, "train")

# 정규화 개선: 표준화 적용
train_images = train_images.reshape(-1, 1, 28, 28) / 255.0
mean = np.mean(train_images)
std = np.std(train_images)
train_images = (train_images - mean) / std

# 데이터 증강 적용
print("데이터 증강 적용 중...")
train_images, train_labels = augment_data(train_images, train_labels)
print(f"데이터 증강 후 데이터 크기: {train_images.shape}")

model = MNIST_CNN()
# 학습 시작
model.train(
    images=train_images,
    labels=train_labels,
    initial_learning_rate=Params.LEARNING_RATE,
    epochs=Params.EPOCHS,
    batch_size=Params.BATCH_SIZE,
)

# 학습한 모델 저장
model.save_weights(CKPT_PATH)
