"""
predict.py - Dự đoán cảm xúc từ ảnh khuôn mặt (đã cắt sẵn từ khung hình).

Tiền xử lý dùng đúng các phép biến đổi của tập val/test trong train.ipynb:
    Grayscale -> Resize -> ToTensor -> Normalize
"""
import os

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from model import MODEL_CONFIG, CLASS_NAMES, build_model


class EmotionPredictor:
    def __init__(self, model_name, weights_path, device="cpu"):
        if model_name not in MODEL_CONFIG:
            raise ValueError(f"Tên mô hình không hợp lệ: {model_name}")
        if not os.path.exists(weights_path):
            raise FileNotFoundError(f"Không tìm thấy file trọng số: {weights_path}")

        self.device = torch.device(device)
        cfg = MODEL_CONFIG[model_name]

        # Tiền xử lý GIỐNG HỆT eval_tf lúc huấn luyện
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=cfg["channels"]),
            transforms.Resize((cfg["img_size"], cfg["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize(cfg["mean"], cfg["std"]),
        ])

        # Nạp file .pth (gồm state_dict và class_names do train.ipynb lưu)
        checkpoint = torch.load(weights_path, map_location=self.device)
        self.class_names = checkpoint.get("class_names", CLASS_NAMES)

        self.model = build_model(model_name, num_classes=len(self.class_names))
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()   # tắt Dropout, BatchNorm dùng thống kê đã học

        # Chạy thử 1 lần cho "nóng máy", tránh khung hình đầu tiên bị giật
        self.predict_batch([np.zeros((64, 64, 3), dtype=np.uint8)])

    def predict_batch(self, faces_bgr):
        """
        Dự đoán cho NHIỀU khuôn mặt cùng lúc (chạy 1 lần qua mạng cho nhanh).
        faces_bgr: danh sách ảnh khuôn mặt (mảng NumPy BGR, như OpenCV đọc ra)
        Trả về: danh sách (tên cảm xúc, độ tin cậy) theo đúng thứ tự đầu vào
        """
        if len(faces_bgr) == 0:
            return []

        tensors = []
        for face in faces_bgr:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)   # BGR -> xám
            pil_img = Image.fromarray(gray)                 # chuyển sang PIL để dùng transforms
            tensors.append(self.transform(pil_img))

        batch = torch.stack(tensors).to(self.device)        # (số mặt, kênh, cao, rộng)

        with torch.no_grad():
            probs = torch.softmax(self.model(batch), dim=1)  # đổi điểm số thành xác suất
        confs, idxs = probs.max(dim=1)                       # lấy lớp có xác suất cao nhất

        return [(self.class_names[i], float(c)) for i, c in zip(idxs.tolist(), confs.tolist())]

    def predict_face(self, face_bgr):
        """Dự đoán cho 1 ảnh khuôn mặt. Trả về (tên cảm xúc, độ tin cậy)."""
        return self.predict_batch([face_bgr])[0]