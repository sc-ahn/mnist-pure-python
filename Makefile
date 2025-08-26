.PHONY: help inspect-data

help: # 도움말 출력
	@echo "Available targets:"
	@grep -E '^[a-zA-Z0-9_-]+:.*#' Makefile | awk -F':.*#' '{printf "  %-15s - %s\n", $$1, $$2}' | sort
	@echo
	@echo "Usage: make <target>"

inspect-data: # dataset의 크기정보를 확인합니다.
	@echo "Dataset size information:"
	uv run python -m src.data

train: # 모델 학습
	uv run python -m src.train

test: # 모델 테스트
	uv run python -m src.test

inference: # 모델 추론 (사용법: make inference img=path/to/image.jpg)
	@if [ -z "$(img)" ]; then echo "Error: img parameter is required. Usage: make inference img=path/to/image.jpg"; exit 1; fi
	uv run python -m inference $(img)

extract-sample: # 샘플 이미지 추출
	uv run python -m src.extract_sample

local-demo: # GUI를 실행합니다.
	$(MAKE) extract-sample
	uv run python -m src.gui

play: # Playground
	uv run python -m playground
