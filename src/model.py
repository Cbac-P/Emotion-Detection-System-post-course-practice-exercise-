"""
model.py - Định nghĩa 2 mô hình, dùng chung cho huấn luyện và chạy realtime.

 - M1 ("cnn")     : CNN tự thiết kế, ảnh xám 1 kênh 48x48
 - M2 ("transfer"): ResNet18 transfer learning, ảnh xám lặp 3 kênh 96x96
"""
import torch.nn as nn
from torchvision import models

# Thứ tự lớp (theo chữ cái, đúng như ImageFolder lúc huấn luyện)
CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Cấu hình tiền xử lý của từng mô hình.
# PHẢI giống hệt lúc huấn luyện trong train.ipynb, nếu khác thì kết quả sẽ sai.
MODEL_CONFIG = {
    "cnn": {
        "img_size": 48, "channels": 1,
        "mean": [0.5], "std": [0.5],
    },
    "transfer": {
        "img_size": 96, "channels": 3,
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],  # chuẩn ImageNet
    },
}


# ---------- M1: CNN tự thiết kế ----------
class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        # Một khối gồm: Conv-BN-ReLU (x2) -> Pooling -> Dropout
        def block(c_in, c_out, drop):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(drop),
            )

        # Ảnh 1x48x48 -> sau 4 lần pooling còn 256x3x3
        self.features = nn.Sequential(
            block(1, 32, 0.2),
            block(32, 64, 0.25),
            block(64, 128, 0.3),
            block(128, 256, 0.3),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ---------- M2: ResNet18 transfer learning ----------
def build_resnet18(num_classes=7, dropout=0.5):
    # weights=None: không cần tải trọng số ImageNet nữa,
    # vì trọng số đã huấn luyện sẽ được nạp từ file .pth
    model = models.resnet18(weights=None)
    # Lớp cuối phải giống hệt lúc train (Dropout + Linear) để nạp state_dict không bị lệch tên
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    return model


def build_model(name, num_classes=7):
    """Tạo mô hình theo tên: 'cnn' hoặc 'transfer'."""
    if name == "cnn":
        return EmotionCNN(num_classes)
    if name == "transfer":
        return build_resnet18(num_classes)
    raise ValueError(f"Tên mô hình không hợp lệ: {name} (chỉ có 'cnn' hoặc 'transfer')")