import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Annotated, Self

import numpy as np
from numpy import dtype, float64, ndarray
from PIL import Image

from src.settings import CKPT_PATH


class ReLU:
    """ReLU 레이어 정의"""

    def forward(self, x: ndarray) -> ndarray:
        """
        ReLU 활성화 함수 적용
        """
        self.input = x
        return np.maximum(0, x)

    def backward(self, d_out: ndarray) -> ndarray:
        """
        ReLU의 역전파
        """
        d_input = d_out.copy()
        d_input[self.input <= 0] = 0  # ReLU의 기울기
        return d_input


class Dropout:
    """
    Dropout 레이어 정의

    Overfitting 방지
    """

    def __init__(self, drop_prob=0.5):
        self.drop_prob = drop_prob
        self.mask = None

    def forward(
        self,
        x: Annotated[ndarray, "입력된 이미지정보"],
        training: Annotated[bool, "훈련 모드 여부"] = True,
    ):
        """
        학습시 랜덤확률로 일부 뉴런의 출력을 0으로 변경
        추론시에는 모든 뉴런의 출력을 그대로 사용
        """
        if not training:
            return x
        self.mask = np.random.rand(*x.shape) > self.drop_prob
        return x * self.mask / (1 - self.drop_prob)  # Scale during training

    def backward(self, d_out):
        """
        forward 에서 사용한 mask를 이용해 역전파
        """
        return d_out * self.mask / (1 - self.drop_prob)


class BatchNorm:
    """
    배치 정규화 레이어 정의

    입력의 평균과 분산을 정규화하여 학습 안정화
    학습 안정화, 더 높은 learning rate 사용, 내부 공변량 이동 감소

    *공변량: 각 층의 입력 분포 변화
    """

    def __init__(self, channels, eps=1e-5):
        self.gamma = np.ones(channels)  # 스케일 파라미터
        self.beta = np.zeros(channels)  # 시프트 파라미터
        self.eps = eps
        self.running_mean = np.zeros(channels)
        self.running_var = np.ones(channels)
        self.momentum = 0.9

    def forward(self, x, training=True):
        if len(x.shape) == 4:  # Conv layer output (N, C, H, W)
            if training:
                # 배치별 평균과 분산 계산
                self.batch_mean = np.mean(x, axis=(0, 2, 3), keepdims=True)
                self.batch_var = np.var(x, axis=(0, 2, 3), keepdims=True)

                # Running statistics 업데이트
                self.running_mean = (
                    self.momentum * self.running_mean
                    + (1 - self.momentum) * self.batch_mean.squeeze()
                )
                self.running_var = (
                    self.momentum * self.running_var
                    + (1 - self.momentum) * self.batch_var.squeeze()
                )

                # 정규화
                self.x_norm = (x - self.batch_mean) / np.sqrt(self.batch_var + self.eps)
            else:
                # 추론 시에는 running statistics 사용
                mean = self.running_mean.reshape(1, -1, 1, 1)
                var = self.running_var.reshape(1, -1, 1, 1)
                self.x_norm = (x - mean) / np.sqrt(var + self.eps)

            # 스케일링과 시프팅
            gamma = self.gamma.reshape(1, -1, 1, 1)
            beta = self.beta.reshape(1, -1, 1, 1)
            return gamma * self.x_norm + beta
        else:  # FC layer output (N, features)
            if training:
                self.batch_mean = np.mean(x, axis=0, keepdims=True)
                self.batch_var = np.var(x, axis=0, keepdims=True)
                self.x_norm = (x - self.batch_mean) / np.sqrt(self.batch_var + self.eps)
            else:
                mean = self.running_mean.reshape(1, -1)
                var = self.running_var.reshape(1, -1)
                self.x_norm = (x - mean) / np.sqrt(var + self.eps)

            return self.gamma * self.x_norm + self.beta

    def backward(self, d_out, learning_rate=0.001):
        if len(d_out.shape) == 4:  # Conv layer
            N, C, H, W = d_out.shape

            # Gradients
            d_gamma = np.sum(d_out * self.x_norm, axis=(0, 2, 3))
            d_beta = np.sum(d_out, axis=(0, 2, 3))

            # Update parameters
            self.gamma -= learning_rate * d_gamma
            self.beta -= learning_rate * d_beta

            # Gradient w.r.t input (simplified)
            gamma = self.gamma.reshape(1, -1, 1, 1)
            d_x = gamma * d_out

        else:  # FC layer
            N = d_out.shape[0]

            d_gamma = np.sum(d_out * self.x_norm, axis=0)
            d_beta = np.sum(d_out, axis=0)

            self.gamma -= learning_rate * d_gamma
            self.beta -= learning_rate * d_beta

            d_x = self.gamma * d_out

        return d_x


# Conv2D 클래스
class Conv2D:
    """
    Convolution 레이어 정의 w/ im2col
    """

    def __init__(
        self,
        input_channels: Annotated[int, "입력 채널 수"],
        output_channels: Annotated[int, "출력 채널 수"],
        kernel_size: Annotated[int, "커널 크기"],
        stride: Annotated[int, "스트라이드 크기"] = 1,
        padding: Annotated[int, "패딩 크기"] = 0,
    ):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel = np.random.randn(
            output_channels, input_channels, kernel_size, kernel_size
        ) * np.sqrt(2.0 / input_channels)
        self.bias = np.zeros(output_channels)

    def _im2col(
        self,
        input_data,
        filter_height,
        filter_width,
        stride,
        padding,
    ):
        N, C, H, W = input_data.shape
        out_height = (H + 2 * padding - filter_height) // stride + 1
        out_width = (W + 2 * padding - filter_width) // stride + 1

        padded_input = np.pad(
            input_data,
            ((0, 0), (0, 0), (padding, padding), (padding, padding)),
            mode="constant",
        )

        col = np.zeros((N, C, filter_height, filter_width, out_height, out_width))
        for y in range(filter_height):
            y_max = y + stride * out_height
            for x in range(filter_width):
                x_max = x + stride * out_width
                col[:, :, y, x, :, :] = padded_input[
                    :, :, y:y_max:stride, x:x_max:stride
                ]

        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_height * out_width, -1)
        return col

    def _col2im(self, col, input_shape, filter_height, filter_width, stride, padding):
        N, C, H, W = input_shape
        out_height = (H + 2 * padding - filter_height) // stride + 1
        out_width = (W + 2 * padding - filter_width) // stride + 1

        col = col.reshape(
            N, out_height, out_width, C, filter_height, filter_width
        ).transpose(0, 3, 4, 5, 1, 2)
        padded_input = np.zeros((N, C, H + 2 * padding, W + 2 * padding))

        for y in range(filter_height):
            y_max = y + stride * out_height
            for x in range(filter_width):
                x_max = x + stride * out_width
                padded_input[:, :, y:y_max:stride, x:x_max:stride] += col[
                    :, :, y, x, :, :
                ]

        return padded_input[:, :, padding : H + padding, padding : W + padding]

    def forward(
        self,
        x: np.ndarray,
    ) -> np.ndarray:
        self.input_shape = x.shape
        N, C, H, W = x.shape
        self.col = self._im2col(
            input_data=x,
            filter_height=self.kernel_size,
            filter_width=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        col_kernel = self.kernel.reshape(self.output_channels, -1).T
        out = np.dot(self.col, col_kernel) + self.bias
        out_height = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        return out.reshape(N, out_height, out_width, self.output_channels).transpose(
            0, 3, 1, 2
        )

    def backward(self, d_out, learning_rate=0.001):
        """
        Backward pass: col2im을 사용해 기울기 계산 및 입력으로 전파
        """
        N, out_channels, out_height, out_width = d_out.shape
        d_out = d_out.transpose(0, 2, 3, 1).reshape(
            -1, out_channels
        )  # (N * out_height * out_width, out_channels)

        # col_kernel은 forward에서 (output_channels, input_channels * kernel_height * kernel_width)로 reshape됨
        col_kernel = self.kernel.reshape(
            self.output_channels, -1
        )  # (out_channels, input_channels * kernel_height * kernel_width)

        # d_col 계산
        d_col = np.dot(
            d_out, col_kernel
        )  # (N * out_height * out_width, input_channels * kernel_height * kernel_width)

        # Gradients for kernel and bias
        grad_kernel = np.dot(
            self.col.T, d_out
        )  # (input_channels * kernel_height * kernel_width, out_channels)
        grad_kernel = grad_kernel.transpose(1, 0).reshape(
            self.kernel.shape
        )  # Reshape to original kernel shape
        grad_bias = np.sum(d_out, axis=0)  # Bias gradient

        # d_input 계산
        d_input = self._col2im(
            d_col,
            self.input_shape,
            self.kernel_size,
            self.kernel_size,
            self.stride,
            self.padding,
        )

        # 파라미터 업데이트
        self.kernel -= learning_rate * grad_kernel
        self.bias -= learning_rate * grad_bias

        return d_input


class Flatten:
    """
    Flatten 레이어 정의

    다차원 텐서 -> 1차원으로 평탄화하는 레이어
    fully connected(dense) 레이어 연결 전 사용
    """

    def forward(self, x):
        """
        입력된 다차원 텐서를 2D로 변환
        """
        self.input_shape = x.shape  # 원본 shape 저장
        return x.reshape(x.shape[0], -1)  # 2D로 변환

    def backward(self, d_out):
        """
        입력된 2D 텐서를 원본 shape으로 변환
        """
        return d_out.reshape(self.input_shape)


class MNIST_CNN:
    """
    MNIST 데이터셋을 위한 CNN 모델 정의
    """

    def __init__(self, num_classes=10, dropout_prob=0.5):
        self.conv1 = Conv2D(1, 8, 3, stride=2, padding=1)  # Conv layer 1
        self.bn1 = BatchNorm(8)  # Batch normalization for conv1
        self.relu1 = ReLU()  # ReLU activation for conv1
        self.conv2 = Conv2D(8, 16, 3, stride=2, padding=1)  # Conv layer 2
        self.bn2 = BatchNorm(16)  # Batch normalization for conv2
        self.relu2 = ReLU()  # ReLU activation for conv2
        self.conv3 = Conv2D(16, 32, 3, stride=2, padding=1)  # Conv layer 3
        self.bn3 = BatchNorm(32)  # Batch normalization for conv3
        self.relu3 = ReLU()  # ReLU activation for conv3
        self.flatten_layer = Flatten()
        self.dropout = Dropout(dropout_prob)  # Dropout layer
        self.num_classes = num_classes

        # Fully connected layer parameters
        self.fc_weights = np.random.randn(32 * 4 * 4, num_classes) * np.sqrt(
            2.0 / (32 * 4 * 4)
        )
        self.fc_bias = np.zeros(num_classes)

    def train(
        self,
        images: Annotated[ndarray[tuple[any, ...], dtype[float64]], "이미지정보"],
        labels: Annotated[ndarray[tuple[any, ...], dtype[any]], "레이블정보"],
        initial_learning_rate: Annotated[float, "학습률"] = 0.01,
        epochs: Annotated[int, "에폭 수"] = 5,
        batch_size: Annotated[int, "배치 크기"] = 128,
    ):
        progress_time_list = []
        for epoch in range(epochs):
            learning_rate = initial_learning_rate * (
                0.9**epoch
            )  # 학습중 학습률 동적 감소
            begin = datetime.now()
            epoch_loss = 0
            num_batches = 0
            for i in range(0, len(images), batch_size):
                batch_images = images[i : i + batch_size]
                batch_labels = labels[i : i + batch_size]

                # Forward Pass, 순전파
                # 예측값 계산과정
                out = self.conv1.forward(batch_images)
                out = self.bn1.forward(out, training=True)
                out = self.relu1.forward(out)
                out = self.conv2.forward(out)
                out = self.bn2.forward(out, training=True)
                out = self.relu2.forward(out)
                out = self.conv3.forward(out)
                out = self.bn3.forward(out, training=True)
                out = self.relu3.forward(out)
                out = self.dropout.forward(out, training=True)
                out = self.flatten_layer.forward(out)
                predictions: ndarray = self.softmax(
                    np.dot(out, self.fc_weights) + self.fc_bias
                )

                # Loss(손실값) 계산
                loss = self.cross_entropy_loss(predictions, batch_labels)
                epoch_loss += loss
                num_batches += 1

                # Backward Pass, 역전파
                # 순전파에서 구한 네트워크 출력과 정답(label)과의 오차를 기준으로
                # 기울기(gradient) 계산하여 파라미터 업데이트

                d_loss = self.cross_entropy_derivative(predictions, batch_labels)
                d_fc = np.dot(out.T, d_loss)
                d_bias = np.sum(d_loss, axis=0)
                d_out = np.dot(d_loss, self.fc_weights.T)

                # Backpropagation through layers
                d_out = self.flatten_layer.backward(d_out)
                d_out = self.dropout.backward(d_out)
                d_out = self.relu3.backward(d_out)
                d_out = self.bn3.backward(d_out, learning_rate)
                d_out = self.conv3.backward(d_out, learning_rate)
                d_out = self.relu2.backward(d_out)
                d_out = self.bn2.backward(d_out, learning_rate)
                d_out = self.conv2.backward(d_out, learning_rate)
                d_out = self.relu1.backward(d_out)
                d_out = self.bn1.backward(d_out, learning_rate)
                d_out = self.conv1.backward(d_out, learning_rate)

                # Update Fully Connected Layer
                self.fc_weights -= learning_rate * d_fc
                self.fc_bias -= learning_rate * d_bias

                print(
                    f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Time: {datetime.now() - begin}",
                    end="\r",
                )

            # 평균 손실 계산
            average_loss = epoch_loss / num_batches
            diff_ = datetime.now() - begin
            diff = diff_.total_seconds()
            progress_time_list.append(diff)
            print(
                f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss:.4f}, Time: {diff:.2f}s"
            )
            print(
                f"Average Time: {sum(progress_time_list) / len(progress_time_list):.2f}s"
            )
        print("Training Finished")

    def evaluate(self, images: ndarray, labels: ndarray) -> float:
        """
        모델 평가 함수
        """
        # Forward Pass
        out = self.conv1.forward(images)
        out = self.bn1.forward(out, training=False)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out, training=False)
        out = self.relu2.forward(out)
        out = self.conv3.forward(out)
        out = self.bn3.forward(out, training=False)
        out = self.relu3.forward(out)
        out = self.flatten_layer.forward(out)
        predictions = self.softmax(np.dot(out, self.fc_weights) + self.fc_bias)

        # Accuracy 계산
        predicted_labels: ndarray = np.argmax(predictions, axis=1)
        accuracy: float = np.mean(predicted_labels == labels)

        print(f"Accuracy: {accuracy * 100:.2f}%")
        return accuracy

    @staticmethod
    def softmax(x: ndarray) -> ndarray:
        """
        소프트맥스, 활성함수
        """
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    @staticmethod
    def cross_entropy_loss(predictions: ndarray, labels: ndarray) -> float:
        m = labels.shape[0]
        log_likelihood = -np.log(predictions[range(m), labels] + 1e-12)
        return np.sum(log_likelihood) / m

    @staticmethod
    def cross_entropy_derivative(predictions: ndarray, labels: ndarray) -> ndarray:
        m = labels.shape[0]
        grad = predictions.copy()
        grad[range(m), labels] -= 1
        return grad / m

    def get_params(self) -> dict:
        return {
            "conv1_kernel": self.conv1.kernel,
            "conv1_bias": self.conv1.bias,
            "bn1_gamma": self.bn1.gamma,
            "bn1_beta": self.bn1.beta,
            "bn1_running_mean": self.bn1.running_mean,
            "bn1_running_var": self.bn1.running_var,
            "conv2_kernel": self.conv2.kernel,
            "conv2_bias": self.conv2.bias,
            "bn2_gamma": self.bn2.gamma,
            "bn2_beta": self.bn2.beta,
            "bn2_running_mean": self.bn2.running_mean,
            "bn2_running_var": self.bn2.running_var,
            "conv3_kernel": self.conv3.kernel,
            "conv3_bias": self.conv3.bias,
            "bn3_gamma": self.bn3.gamma,
            "bn3_beta": self.bn3.beta,
            "bn3_running_mean": self.bn3.running_mean,
            "bn3_running_var": self.bn3.running_var,
            "fc_weights": self.fc_weights,
            "fc_bias": self.fc_bias,
            "num_classes": self.num_classes,
        }

    def save_weights(self, filepath: Path):
        """
        학습된 모델의 가중치를 파일로 저장
        """
        weights_dict = self.get_params()

        # 디렉토리가 없으면 생성
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "wb") as f:
            pickle.dump(weights_dict, f)
        print(f"모델 가중치가 {filepath}에 저장되었습니다.")

    @staticmethod
    def load_weights(filepath: Path) -> "MNIST_CNN":
        """
        저장된 가중치를 불러와서 모델에 적용
        """

        if not filepath.exists():
            raise FileNotFoundError(f"가중치 파일을 찾을 수 없습니다: {filepath}")

        with open(filepath, "rb") as f:
            weights = pickle.load(f)

        model = MNIST_CNN(
            num_classes=weights["num_classes"],
        )
        model.conv1.kernel = weights["conv1_kernel"]
        model.conv1.bias = weights["conv1_bias"]
        model.bn1.gamma = weights.get("bn1_gamma", np.ones(8))
        model.bn1.beta = weights.get("bn1_beta", np.zeros(8))
        model.bn1.running_mean = weights.get("bn1_running_mean", np.zeros(8))
        model.bn1.running_var = weights.get("bn1_running_var", np.ones(8))
        model.conv2.kernel = weights["conv2_kernel"]
        model.conv2.bias = weights["conv2_bias"]
        model.bn2.gamma = weights.get("bn2_gamma", np.ones(16))
        model.bn2.beta = weights.get("bn2_beta", np.zeros(16))
        model.bn2.running_mean = weights.get("bn2_running_mean", np.zeros(16))
        model.bn2.running_var = weights.get("bn2_running_var", np.ones(16))
        model.conv3.kernel = weights["conv3_kernel"]
        model.conv3.bias = weights["conv3_bias"]
        model.bn3.gamma = weights.get("bn3_gamma", np.ones(32))
        model.bn3.beta = weights.get("bn3_beta", np.zeros(32))
        model.bn3.running_mean = weights.get("bn3_running_mean", np.zeros(32))
        model.bn3.running_var = weights.get("bn3_running_var", np.ones(32))
        model.fc_weights = weights["fc_weights"]
        model.fc_bias = weights["fc_bias"]
        return model

    def predict_with_ndarray(self, image: ndarray) -> tuple[int, float]:
        """
        이미지 배열을 입력받아서 예측 결과 반환
        """
        # 전처리
        image = image.reshape(1, 1, 28, 28)
        # 예측
        out = self.conv1.forward(image)
        out = self.bn1.forward(out, training=False)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out, training=False)
        out = self.relu2.forward(out)
        out = self.conv3.forward(out)
        out = self.bn3.forward(out, training=False)
        out = self.relu3.forward(out)
        out = self.flatten_layer.forward(out)
        predictions = self.softmax(np.dot(out, self.fc_weights) + self.fc_bias)

        # 결과 반환
        predicted_label = np.argmax(predictions[0])
        confidence = predictions[0][predicted_label]
        return int(predicted_label), float(confidence)

    def predict_with_file(self, image_path: str) -> tuple[int, float]:
        """
        이미지 파일을 입력받아서 예측 결과 반환
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")

        # 이미지 로드 및 전처리
        img = Image.open(image_path)

        # 그레이스케일 변환
        if img.mode != "L":
            img = img.convert("L")

        # 28x28로 리사이즈
        img = img.resize((28, 28), Image.Resampling.LANCZOS)

        # numpy 배열로 변환 및 정규화 (0-1 범위)
        img_array = np.array(img, dtype=np.float32) / 255.0

        # 배치 차원 추가 (1, 1, 28, 28)
        img_array = img_array.reshape(1, 1, 28, 28)

        # 예측 수행
        out = self.conv1.forward(img_array)
        out = self.bn1.forward(out, training=False)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        out = self.bn2.forward(out, training=False)
        out = self.relu2.forward(out)
        out = self.conv3.forward(out)
        out = self.bn3.forward(out, training=False)
        out = self.relu3.forward(out)
        out = self.flatten_layer.forward(out)
        predictions = self.softmax(np.dot(out, self.fc_weights) + self.fc_bias)

        # 예측 결과
        predicted_label = np.argmax(predictions[0])
        confidence = predictions[0][predicted_label]

        print(f"예측 결과: {predicted_label}, 확신도: {confidence:.4f}")
        return int(predicted_label), float(confidence)


def main(data: np.ndarray):
    model = MNIST_CNN.load_weights(CKPT_PATH)
    pred = model.predict_with_ndarray(data)
    return pred


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
