"""
모든 논문 구현에서 공통으로 사용하는 유틸리티 함수.
학습 결과를 자동으로 results/ 폴더에 저장한다.
"""
import os
import csv
import json
from datetime import datetime

import torch
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False


def get_device():
    """학습 디바이스 자동 감지"""
    return torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def ensure_dir(path):
    """폴더가 없으면 생성"""
    os.makedirs(path, exist_ok=True)


class ResultLogger:
    """
    학습 결과를 자동으로 저장하는 로거.

    사용 예시:
        logger = ResultLogger("results", model_name="LeNet")
        logger.log(epoch=1, train_loss=0.1, train_acc=0.95, test_loss=0.2, test_acc=0.93)
        ...
        logger.save_model(model)
        logger.save_curves()
        logger.save_summary(config={...})
    """
    def __init__(self, results_dir, model_name="model"):
        self.results_dir = results_dir
        self.model_name = model_name
        ensure_dir(results_dir)

        self.history = []
        self.start_time = datetime.now()

    def log(self, **kwargs):
        """에폭마다 결과 기록"""
        self.history.append(kwargs)

    def save_csv(self, filename="training_log.csv"):
        """학습 로그를 CSV로 저장"""
        if not self.history:
            return
        path = os.path.join(self.results_dir, filename)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.history[0].keys())
            writer.writeheader()
            writer.writerows(self.history)
        print(f"학습 로그 저장: {path}")

    def save_curves(self, filename="training_curve.png"):
        """학습 곡선을 그래프로 저장"""
        if not self.history:
            return
        path = os.path.join(self.results_dir, filename)

        epochs = [h.get("epoch", i+1) for i, h in enumerate(self.history)]

        # Loss와 Accuracy 모두 있는지 확인
        has_loss = "train_loss" in self.history[0]
        has_acc = "train_acc" in self.history[0]

        if has_loss and has_acc:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))

            # Loss 곡선
            train_loss = [h["train_loss"] for h in self.history]
            axes[0].plot(epochs, train_loss, label="Train")
            if "test_loss" in self.history[0]:
                test_loss = [h["test_loss"] for h in self.history]
                axes[0].plot(epochs, test_loss, label="Test")
            axes[0].set_xlabel("Epoch")
            axes[0].set_ylabel("Loss")
            axes[0].set_title(f"{self.model_name} - Loss")
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)

            # Accuracy 곡선
            train_acc = [h["train_acc"] for h in self.history]
            axes[1].plot(epochs, train_acc, label="Train")
            if "test_acc" in self.history[0]:
                test_acc = [h["test_acc"] for h in self.history]
                axes[1].plot(epochs, test_acc, label="Test")
            axes[1].set_xlabel("Epoch")
            axes[1].set_ylabel("Accuracy")
            axes[1].set_title(f"{self.model_name} - Accuracy")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(path, dpi=120)
            plt.close()
        elif has_loss:
            plt.figure(figsize=(8, 4))
            train_loss = [h["train_loss"] for h in self.history]
            plt.plot(epochs, train_loss, label="Train")
            if "test_loss" in self.history[0]:
                test_loss = [h["test_loss"] for h in self.history]
                plt.plot(epochs, test_loss, label="Test")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title(f"{self.model_name} - Loss")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(path, dpi=120)
            plt.close()

        print(f"학습 곡선 저장: {path}")

    def save_model(self, model, filename=None):
        """모델 가중치 저장"""
        if filename is None:
            filename = f"{self.model_name.lower()}.pth"
        path = os.path.join(self.results_dir, filename)
        torch.save(model.state_dict(), path)
        print(f"모델 저장: {path}")

    def save_summary(self, config=None, extra=None):
        """학습 요약을 markdown으로 저장"""
        path = os.path.join(self.results_dir, "summary.md")
        end_time = datetime.now()
        elapsed = end_time - self.start_time

        with open(path, "w") as f:
            f.write(f"# {self.model_name} 학습 결과\n\n")
            f.write(f"- 학습 시작: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- 학습 종료: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"- 소요 시간: {elapsed}\n\n")

            if config:
                f.write("## 학습 설정\n\n")
                f.write("| 항목 | 값 |\n")
                f.write("| --- | --- |\n")
                for k, v in config.items():
                    f.write(f"| {k} | {v} |\n")
                f.write("\n")

            if self.history:
                f.write("## 학습 로그\n\n")
                keys = list(self.history[0].keys())
                f.write("| " + " | ".join(keys) + " |\n")
                f.write("| " + " | ".join(["---"] * len(keys)) + " |\n")
                for row in self.history:
                    values = []
                    for k in keys:
                        v = row[k]
                        if isinstance(v, float):
                            values.append(f"{v:.4f}")
                        else:
                            values.append(str(v))
                    f.write("| " + " | ".join(values) + " |\n")
                f.write("\n")

                # 최종 성능
                last = self.history[-1]
                f.write("## 최종 성능\n\n")
                for k, v in last.items():
                    if isinstance(v, float):
                        f.write(f"- {k}: {v:.4f}\n")
                    else:
                        f.write(f"- {k}: {v}\n")
                f.write("\n")

            if extra:
                f.write("## 추가 정보\n\n")
                f.write(extra + "\n")

        print(f"학습 요약 저장: {path}")

    def save_all(self, model=None, config=None):
        """모든 결과를 한번에 저장"""
        self.save_csv()
        self.save_curves()
        if model is not None:
            self.save_model(model)
        self.save_summary(config=config)
