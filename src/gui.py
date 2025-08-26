import tkinter as tk

import numpy as np
from PIL import Image

from src.model import MNIST_CNN
from src.settings import CKPT_PATH, IMG_DIR


def main():
    print("loading model...")
    model = MNIST_CNN.load_weights(CKPT_PATH)
    print("model loaded")

    root = tk.Tk()
    root.title("MNIST Image Viewer")
    root.geometry("800x400")

    # 1. 왼쪽에 파일 목록을 띄워주는 리스트 박스 생성하고 파일 목록을 띄워줌
    # 스크롤바와 함께 리스트박스를 위한 프레임 생성
    left_frame = tk.Frame(root)
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # 스크롤바 생성
    scrollbar = tk.Scrollbar(left_frame)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    # 리스트박스 생성 및 스크롤바 연결
    listbox = tk.Listbox(left_frame, yscrollcommand=scrollbar.set)
    listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.config(command=listbox.yview)

    # 파일 경로 받아옴
    for image_path in IMG_DIR.iterdir():
        # 만약 파일이 아니라면 넘어감
        if not image_path.is_file():
            continue
        listbox.insert(tk.END, image_path.name)

    # 1-1. 파일 목록에 있는 파일을 클릭하면 오른쪽에 이미지를 띄워줌
    pred_label = None  # 예측 정보 레이블을 저장할 변수

    def on_select(event):
        nonlocal pred_label
        index = listbox.curselection()
        if index:
            print(listbox.get(index))
            image_path = IMG_DIR / listbox.get(index)
            print(image_path, type(image_path))

            # 기존 예측 정보 레이블이 있으면 삭제
            if pred_label is not None:
                pred_label.destroy()
                pred_label = None

            # 이미지를 띄워줌
            image = tk.PhotoImage(file=image_path)
            image_label.config(image=image)
            image_label.image = image  # type: ignore

            image = Image.open(image_path).convert("L")  # Grayscale로 변환
            image = image.resize((28, 28))  # 모델 입력 크기에 맞게 조정
            image_np = np.array(image).reshape(1, 1, 28, 28) / 255.0

            pred, conf_ = model.predict_with_ndarray(image_np)
            # 소숫점 2자리로 반올림
            conf = round(conf_ * 100, 2)
            print(f"Prediction: {pred}, Confidence: {conf:.2f}%")

            # 이미지를 GUI에 표시
            tk_image = tk.PhotoImage(file=image_path)
            image_label.config(image=tk_image)
            image_label.image = tk_image  # type: ignore

            # 예측된 prediction을 GUI에 표시 (새로 생성)
            pred_label = tk.Label(
                root,
                text=f"Prediction: {pred}, Confidence: {conf:.2f}%",
                font=("Helvetica", 16),
            )
            pred_label.pack(side=tk.BOTTOM)

    listbox.bind("<<ListboxSelect>>", on_select)

    # 2. 이미지를 띄워주는 레이블 생성하고 이미지를 띄워줌
    image_label = tk.Label(root)
    image_label.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

    root.mainloop()


if __name__ == "__main__":
    main()
