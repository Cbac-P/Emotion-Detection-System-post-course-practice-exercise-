"""
detector.py - Phát hiện khuôn mặt bằng Haar Cascade của OpenCV.

Haar Cascade có sẵn trong thư viện OpenCV (không cần tải thêm file),
chạy nhanh trên CPU nên dễ đạt >= 10 FPS.
"""
import cv2


class FaceDetector:
    def __init__(self, scale_factor=1.1, min_neighbors=5, min_face_size=60, detect_width=640,
                 min_detect_width=256):
        """
        scale_factor : mức thu nhỏ ảnh sau mỗi vòng quét (nhỏ -> chính xác hơn nhưng chậm hơn)
        min_neighbors: số ô lân cận tối thiểu để chấp nhận 1 khuôn mặt (lớn -> ít nhận nhầm hơn)
        min_face_size: kích thước mặt nhỏ nhất (pixel) tính trên ảnh đã thu nhỏ
        detect_width : ảnh rộng hơn mức này sẽ được thu nhỏ trước khi dò mặt cho nhanh
        min_detect_width: ảnh hẹp hơn mức này sẽ được PHÓNG TO trước khi dò mặt
                          (ảnh nhỏ như FER-2013 48x48 nhỏ hơn cả min_face_size nên không dò được)
        """
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(path)
        if self.cascade.empty():
            raise RuntimeError(f"Không nạp được file Haar Cascade: {path}")

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_face_size = min_face_size
        self.detect_width = detect_width
        self.min_detect_width = min_detect_width

    def _find(self, gray):
        """Chạy Haar trên 1 ảnh xám, trả về các khung (x, y, w, h) theo tọa độ của ảnh đó."""
        return self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_face_size, self.min_face_size),
        )

    def detect(self, frame_bgr):
        """Trả về danh sách khuôn mặt [(x, y, w, h), ...] theo tọa độ ảnh GỐC."""
        h, w = frame_bgr.shape[:2]

        # Đưa ảnh về kích thước hợp lý trước khi dò mặt:
        #  - ảnh quá lớn (camera điện thoại 1080p): thu nhỏ cho dò nhanh hơn
        #  - ảnh quá nhỏ (ví dụ ảnh 48x48 của FER-2013): phóng to, vì cửa sổ dò nhỏ nhất
        #    là min_face_size pixel, ảnh nhỏ hơn mức đó thì không bao giờ dò được
        scale = 1.0
        small = frame_bgr
        if w > self.detect_width:
            scale = self.detect_width / w
            small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        elif w < self.min_detect_width:
            scale = self.min_detect_width / w
            small = cv2.resize(frame_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Haar làm việc trên ảnh xám; cân bằng histogram giúp ổn định khi thiếu sáng
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        # Lần 1: dò trên ảnh nguyên bản (nhanh, đủ cho hầu hết khung hình từ camera)
        boxes = self._find(gray)
        pad = 0

        # Lần 2: nếu không thấy mặt nào thì thêm viền đen quanh ảnh rồi dò lại.
        # Lý do: Haar cần có "khoảng trống" quanh mặt; ảnh cận mặt (mặt choán gần hết
        # khung hình) sẽ bị bỏ sót nếu không có viền.
        if len(boxes) == 0:
            pad = int(0.15 * max(gray.shape))
            padded = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            boxes = self._find(padded)

        # Đổi tọa độ về ảnh gốc: trừ viền, chia tỉ lệ thu nhỏ, rồi cắt cho nằm trong ảnh
        result = []
        for (x, y, bw, bh) in boxes:
            x1 = max(int((x - pad) / scale), 0)
            y1 = max(int((y - pad) / scale), 0)
            x2 = min(int((x - pad + bw) / scale), w)
            y2 = min(int((y - pad + bh) / scale), h)
            if x2 > x1 and y2 > y1:
                result.append((x1, y1, x2 - x1, y2 - y1))
        return result