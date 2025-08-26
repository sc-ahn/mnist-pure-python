import matplotlib.patches as patches
import matplotlib.pyplot as plt


# 모델 구조 시각화 함수
def draw_mnist_cnn_model():
    fig, ax = plt.subplots(figsize=(12, 6))

    # 배경 제거
    ax.axis("off")

    # 각 레이어 박스 위치 설정 (x, y, width, height)
    layers = [
        ("Input\n(1x28x28)", 0, 2, 2),
        ("Conv2D\n(8x14x14)", 2.5, 2, 2),
        ("Conv2D\n(16x7x7)", 5, 2, 2),
        ("Conv2D\n(32x4x4)", 7.5, 2, 2),
        ("Dropout", 10, 2, 2),
        ("Flatten\n(512)", 12.5, 2, 2),
        ("FC\n(10)", 15, 2, 2),
        ("Softmax\nOutput", 17.5, 2, 2),
    ]

    for name, x, w, h in layers:
        rect = patches.FancyBboxPatch(
            (x, 1),
            w,
            h,
            boxstyle="round,pad=0.02",
            edgecolor="black",
            facecolor="#87CEFA",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(
            x + w / 2, 2, name, ha="center", va="center", fontsize=10, weight="bold"
        )

    plt.xlim(-1, 21)
    plt.ylim(0.5, 4)
    plt.tight_layout()
    plt.show()


draw_mnist_cnn_model()
