"""
app.py - Nhận diện cảm xúc khuôn mặt thời gian thực.

Cách chạy (nhấn phím q để thoát):
    python src/app.py --source http://<ip>:8080/video --model cnn       # IP camera điện thoại
    python src/app.py --source test.jpg --model transfer                # ảnh tĩnh
    python src/app.py --source video.mp4 --model cnn                    # file video
    python src/app.py --source 0 --model cnn                            # webcam laptop
"""
import argparse
import os

# Tránh lỗi OMP #15 do trùng thư viện OpenMP giữa Anaconda và PyTorch trên Windows
# (phải đặt TRƯỚC khi import torch / numpy)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from detector import FaceDetector
from predict import EmotionPredictor

ROOT = Path(__file__).resolve().parent.parent          # thư mục gốc của dự án
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
COLOR_OK = (0, 255, 0)        # xanh lá: độ tin cậy đủ cao   (OpenCV dùng thứ tự BGR)
COLOR_LOW = (0, 165, 255)     # cam: độ tin cậy thấp
WINDOW = "Emotion"
MAX_WIDTH = 700      # chiều rộng tối đa của khung hình hiển thị (pixel), chỉnh số này để to/nhỏ
MIN_DISPLAY_WIDTH = 480     # ảnh tĩnh nhỏ hơn mức này sẽ được phóng to để hiển thị cho rõ
# ---------------------------------------------------------------
# 1. Xử lý 1 khung hình: dò mặt -> dự đoán -> vẽ kết quả
# ---------------------------------------------------------------
def shrink(frame, max_width):
    """Thu nhỏ khung hình nếu rộng hơn max_width (giữ nguyên tỉ lệ). Ảnh nhỏ hơn thì để nguyên."""
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    return cv2.resize(frame, None, fx=scale, fy=scale)


def analyze(frame, detector, predictor):
    """Dò mặt rồi dự đoán cảm xúc. Trả về (danh sách khung, danh sách (nhãn, confidence))."""
    # Dò khuôn mặt, cắt từng vùng mặt ra
    faces, boxes = [], []
    for (x, y, w, h) in detector.detect(frame):
        face = frame[y:y + h, x:x + w]
        if face.size == 0:          # vùng cắt rỗng thì bỏ qua
            continue
        faces.append(face)
        boxes.append((x, y, w, h))

    # Dự đoán tất cả khuôn mặt trong 1 lần chạy
    results = predictor.predict_batch(faces)
    return boxes, results


def draw_results(frame, boxes, results, threshold):
    """Vẽ bounding box + nhãn cảm xúc + confidence lên frame."""
    frame_h = frame.shape[0]
    for (x, y, w, h), (label, conf) in zip(boxes, results):
        color = COLOR_OK if conf >= threshold else COLOR_LOW
        text = f"{label} {conf * 100:.0f}%"
        if conf < threshold:
            text += " ?"            # dấu ? báo hiệu mô hình chưa chắc chắn

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Chọn vị trí chữ: ưu tiên phía trên khung; không đủ chỗ thì xuống dưới;
        # nếu dưới cũng không đủ chỗ (mặt choán gần hết ảnh) thì ghi vào bên trong khung
        if y > 25:
            y_text = y - 10
        elif y + h + 25 < frame_h:
            y_text = y + h + 25
        else:
            y_text = y + 25
        cv2.putText(frame, text, (max(x, 0), y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)


def process_frame(frame, detector, predictor, threshold):
    """Dò mặt + dự đoán + vẽ kết quả lên frame. Trả về (frame, số mặt)."""
    boxes, results = analyze(frame, detector, predictor)
    draw_results(frame, boxes, results, threshold)
    return frame, len(boxes)


def draw_info(frame, fps, num_faces):
    """Ghi FPS và số khuôn mặt ở góc trên bên trái."""
    text = f"FPS: {fps:.1f} | Faces: {num_faces}"
    cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)


def make_message_frame(text):
    """Tạo khung đen có dòng chữ (dùng khi chưa có hình từ camera)."""
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(canvas, text, (30, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    return canvas


# ---------------------------------------------------------------
# 2. Đọc luồng camera bằng thread riêng (giảm độ trễ, tự kết nối lại)
# ---------------------------------------------------------------
class StreamReader:
    """
    Một thread chạy nền liên tục đọc camera và chỉ giữ khung hình MỚI NHẤT.
    Nhờ vậy hình hiển thị không bị trễ dù mô hình xử lý chậm hơn camera.
    Nếu mất kết nối, thread tự thử kết nối lại, chương trình chính không bị crash.
    """

    def __init__(self, source):
        self.source = source
        self.cap = None
        self.frame = None
        self.frame_id = 0               # tăng 1 mỗi khi có khung hình mới
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            # Bước 1: mở kết nối nếu chưa có
            if self.cap is None:
                cap = cv2.VideoCapture(self.source)
                if not cap.isOpened():
                    cap.release()
                    print("Không kết nối được camera, thử lại sau 2 giây...")
                    time.sleep(2)
                    continue
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # hạn chế khung hình bị dồn ứ
                self.cap = cap
                print("Đã kết nối camera.")

            # Bước 2: đọc 1 khung hình
            ok, frame = self.cap.read()
            if not ok:
                print("Mất kết nối camera, đang thử lại...")
                self.cap.release()
                self.cap = None
                with self.lock:
                    self.frame = None
                time.sleep(1)
                continue

            with self.lock:
                self.frame = frame
                self.frame_id += 1

    def read(self):
        """Trả về (số thứ tự khung hình, khung hình). Khung hình = None nếu chưa có / đang mất kết nối."""
        with self.lock:
            if self.frame is None:
                return self.frame_id, None
            return self.frame_id, self.frame.copy()

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()


# ---------------------------------------------------------------
# 3. Ba chế độ chạy: ảnh tĩnh / file video / luồng camera
# ---------------------------------------------------------------
def run_image(path, detector, predictor, threshold, output):
    # Đọc ảnh bằng imdecode để đường dẫn có dấu tiếng Việt vẫn đọc được trên Windows
    data = np.fromfile(path, dtype=np.uint8)
    frame = cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size > 0 else None
    if frame is None:
        sys.exit(f"Lỗi: không đọc được ảnh '{path}' (file hỏng hoặc không phải ảnh).")

    # Dự đoán trên ảnh GỐC (để kết quả giống lúc huấn luyện)
    boxes, results = analyze(frame, detector, predictor)

    # Ảnh nhỏ (ví dụ FER-2013 48x48) thì phóng to lên để vẽ khung và chữ cho rõ.
    # Phải phóng to SAU khi dự đoán, rồi nhân tọa độ khung với cùng tỉ lệ.
    h, w = frame.shape[:2]
    if w < MIN_DISPLAY_WIDTH:
        k = MIN_DISPLAY_WIDTH / w
        frame = cv2.resize(frame, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC)
        boxes = [(int(x * k), int(y * k), int(bw * k), int(bh * k)) for (x, y, bw, bh) in boxes]

    draw_results(frame, boxes, results, threshold)

    num_faces = len(boxes)
    print(f"Phát hiện {num_faces} khuôn mặt.")
    if num_faces == 0:
        print("Không phát hiện khuôn mặt nào trong ảnh.")
    for label, conf in results:
        print(f" - {label}: {conf * 100:.1f}%")
        

    if output:      # lưu ảnh kết quả nếu có yêu cầu
        ext = os.path.splitext(output)[1] or ".jpg"
        cv2.imencode(ext, frame)[1].tofile(output)
        print("Đã lưu ảnh kết quả:", output)

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)      # cho phép kéo to/nhỏ cửa sổ
    cv2.imshow(WINDOW, frame)
    print("Nhấn phím bất kỳ trong cửa sổ ảnh để thoát.")
    cv2.waitKey(0)


def run_video_file(path, detector, predictor, threshold):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Lỗi: không mở được file video '{path}'.")

    fps, t_prev = 0.0, time.time()
    while True:
        ok, frame = cap.read()
        if not ok:                      # hết video
            print("Đã xử lý hết video.")
            break

        frame = shrink(frame, MAX_WIDTH)          # thu nhỏ khung hình cho vừa màn hình    
        frame, num_faces = process_frame(frame, detector, predictor, threshold)

        # Tính FPS (làm mượt để số không nhảy lung tung)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6) if fps else 1 / max(now - t_prev, 1e-6)
        t_prev = now

        draw_info(frame, fps, num_faces)
        cv2.imshow(WINDOW, frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()


def run_stream(source, detector, predictor, threshold):
    reader = StreamReader(source)
    last_id, fps, t_prev = -1, 0.0, time.time()
    print("Đang kết nối camera... (nhấn q để thoát)")

    while True:
        frame_id, frame = reader.read()

        if frame is None:
            # Chưa có hình hoặc đang mất kết nối: hiện thông báo, không crash
            cv2.imshow(WINDOW, make_message_frame("Camera disconnected - reconnecting..."))
            key = cv2.waitKey(200)
        elif frame_id == last_id:
            # Chưa có khung hình mới thì chờ một chút, không xử lý lại hình cũ
            key = cv2.waitKey(5)
        else:
            last_id = frame_id
            frame = shrink(frame, MAX_WIDTH)      # thu nhỏ khung hình cho vừa màn hình
            frame, num_faces = process_frame(frame, detector, predictor, threshold)

            now = time.time()
            fps = 0.9 * fps + 0.1 / max(now - t_prev, 1e-6) if fps else 1 / max(now - t_prev, 1e-6)
            t_prev = now

            draw_info(frame, fps, num_faces)
            cv2.imshow(WINDOW, frame)
            key = cv2.waitKey(1)

        if key & 0xFF == ord("q"):
            break

    reader.stop()


# ---------------------------------------------------------------
# 4. Đọc tham số + cấu hình rồi chạy
# ---------------------------------------------------------------
def get_source_type(source):
    """Phân loại nguồn: webcam (số) / stream (URL) / image / video."""
    if source.isdigit():
        return "webcam"
    if source.lower().startswith(("http://", "https://", "rtsp://")):
        return "stream"
    if os.path.splitext(source)[1].lower() in IMAGE_EXTS:
        return "image"
    return "video"


def main():
    parser = argparse.ArgumentParser(description="Nhận diện cảm xúc khuôn mặt realtime")
    parser.add_argument("--source", default=None,
                        help="URL IP camera, đường dẫn ảnh/video, hoặc số webcam (0). "
                             "Bỏ trống thì lấy camera.source trong config.yaml")
    parser.add_argument("--model", choices=["cnn", "transfer"], default="cnn",
                        help="cnn = M1 (CNN tự thiết kế), transfer = M2 (ResNet18)")
    parser.add_argument("--config", default=str(ROOT / "config.yaml"), help="đường dẫn file config.yaml")
    parser.add_argument("--threshold", type=float, default=None,
                        help="ngưỡng confidence (ghi đè giá trị trong config)")
    parser.add_argument("--device", default="cpu", help="cpu hoặc cuda (mặc định cpu)")
    parser.add_argument("--output", default=None, help="đường dẫn lưu ảnh kết quả (chỉ dùng với ảnh tĩnh)")
    args = parser.parse_args()

    # Đọc file cấu hình
    try:
        with open(args.config, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        sys.exit(f"Lỗi: không tìm thấy file cấu hình '{args.config}'.")

    # Tham số dòng lệnh được ưu tiên hơn file cấu hình
    source = args.source or cfg.get("camera", {}).get("source")
    if not source:
        sys.exit("Lỗi: chưa có nguồn video. Hãy dùng --source hoặc điền camera.source trong config.yaml.")
    source = str(source)
    threshold = args.threshold if args.threshold is not None else cfg.get("confidence_threshold", 0.5)

    # Đường dẫn trọng số: tính từ thư mục gốc dự án nếu là đường dẫn tương đối
    weights_path = Path(cfg["models"][args.model])
    if not weights_path.is_absolute():
        weights_path = ROOT / weights_path

    # Khởi tạo bộ dò mặt và bộ dự đoán
    try:
        det_cfg = cfg.get("detector", {})
        detector = FaceDetector(
            scale_factor=det_cfg.get("scale_factor", 1.1),
            min_neighbors=det_cfg.get("min_neighbors", 5),
            min_face_size=det_cfg.get("min_face_size", 60),
            detect_width=det_cfg.get("detect_width", 640),
        )
        predictor = EmotionPredictor(args.model, str(weights_path), device=args.device)
    except (FileNotFoundError, RuntimeError, ValueError) as e:
        sys.exit(f"Lỗi khởi tạo: {e}")

    print(f"Mô hình: {args.model} | Nguồn: {source} | Ngưỡng confidence: {threshold}")

    # Chạy theo loại nguồn
    source_type = get_source_type(source)
    try:
        if source_type == "image":
            run_image(source, detector, predictor, threshold, args.output)
        elif source_type == "video":
            run_video_file(source, detector, predictor, threshold)
        else:   # webcam hoặc stream
            run_stream(int(source) if source_type == "webcam" else source,
                       detector, predictor, threshold)
    except KeyboardInterrupt:
        print("\nĐã dừng bằng Ctrl+C.")
    finally:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()