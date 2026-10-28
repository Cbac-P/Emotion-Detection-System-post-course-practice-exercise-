# Lab 2 – Nhận diện cảm xúc khuôn mặt bằng Deep Learning

**Sinh viên:** Nguyễn Thanh Phương – **MSSV:** CE200567
**Giảng viên:** TS. Nguyễn Đình Vinh

Hệ thống nhận diện cảm xúc khuôn mặt thời gian thực: đọc luồng video từ camera điện thoại (IP camera), phát hiện khuôn mặt, phân loại 7 cảm xúc và hiển thị khung, nhãn, độ tin cậy (confidence) cùng FPS lên khung hình.

7 lớp cảm xúc: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.

## 1. Cấu trúc thư mục

```
CE200567_NguyenThanhPhuong_Lab2/
├── README.md
├── requirements.txt
├── config.yaml              # URL camera, đường dẫn model, ngưỡng confidence
├── notebooks/
│   └── train.ipynb          # huấn luyện + đánh giá M1, M2 (đã chạy, còn output)
├── src/
│   ├── model.py             # định nghĩa mô hình M1, M2 và cấu hình tiền xử lý
│   ├── detector.py          # phát hiện khuôn mặt (Haar Cascade)
│   ├── predict.py           # dự đoán cảm xúc cho ảnh khuôn mặt
│   └── app.py               # ứng dụng realtime (--source, --model)
├── weights/
│   ├── m1_cnn.pth           # trọng số M1 (CNN tự thiết kế)
│   └── m2_resnet18.pth      # trọng số M2 (ResNet18 transfer learning)
└── report.pdf               # báo cáo
```

## 2. Cài đặt

Yêu cầu: Python 3.9 trở lên, laptop có CPU (không bắt buộc GPU để chạy ứng dụng).

```bash
pip install -r requirements.txt
```

Lưu ý khi cài đặt:

- Dùng `opencv-python` **bản 4.x** (đã ghi `opencv-python<5` trong `requirements.txt`). Bản 5.x không còn `CascadeClassifier` nên bộ dò mặt sẽ không chạy.
- Trên Windows + Anaconda, nếu gặp lỗi `OMP: Error #15`, đặt biến môi trường `KMP_DUPLICATE_LIB_OK=TRUE` (trong `app.py` đã đặt sẵn).
- Trọng số nằm trong thư mục `weights/`.

## 3. Cách chạy ứng dụng

Hai lệnh bắt buộc theo đề:

```bash
python src/app.py --source http://<ip>:8080/video --model cnn
python src/app.py --source test.jpg --model transfer
```

Hãy thay `<ip>` bằng địa chỉ IP thật của điện thoại (ví dụ `192.168.1.15`).

Các ví dụ khác:

```bash
python src/app.py --source video.mp4 --model cnn      # file video
python src/app.py --source 0 --model cnn              # webcam laptop
python src/app.py --source test.jpg --model transfer --output ketqua.jpg   # lưu ảnh kết quả
```

### Tham số dòng lệnh

| Tham số | Ý nghĩa | Mặc định |
|---|---|---|
| `--source` | URL IP camera, đường dẫn ảnh/video, hoặc số webcam (`0`) | lấy `camera.source` trong `config.yaml` |
| `--model` | `cnn` = M1 (CNN tự thiết kế), `transfer` = M2 (ResNet18) | `cnn` |
| `--config` | đường dẫn file cấu hình | `config.yaml` |
| `--threshold` | ngưỡng confidence (ghi đè giá trị trong config) | theo `config.yaml` (0.5) |
| `--device` | `cpu` hoặc `cuda` | `cpu` |
| `--output` | đường dẫn lưu ảnh kết quả (chỉ với ảnh tĩnh) | không lưu |

Thoát chương trình: nhấn **q** (video/camera) hoặc phím bất kỳ (ảnh tĩnh).

### Kết nối IP camera điện thoại

1. Cài ứng dụng IP Webcam (Android) hoặc DroidCam / iVCam / EpocCam.
2. Mở ứng dụng, chọn **Start server**, ghi lại địa chỉ hiển thị (ví dụ `http://192.168.1.15:8080`).
3. Điện thoại và laptop phải cùng một mạng Wi-Fi (hoặc laptop bắt hotspot của điện thoại). Kiểm tra bằng cách mở địa chỉ đó trên trình duyệt của laptop.
4. Chạy ứng dụng với `--source http://<ip>:8080/video` (DroidCam dùng cổng `4747`).

Địa chỉ camera **không được ghi cứng trong mã nguồn**. Có hai cách thay đổi khi đổi mạng:

- Dùng tham số `--source http://<ip-mới>:8080/video`, hoặc
- Sửa `camera.source` trong `config.yaml`.

Tham số dòng lệnh luôn được ưu tiên hơn file cấu hình.

### File cấu hình `config.yaml`

```yaml
camera:
  source: "http://192.168.1.15:8080/video"   # địa chỉ mẫu, sửa theo IP hiện tại hoặc dùng --source
models:
  cnn: weights/m1_cnn.pth
  transfer: weights/m2_resnet18.pth
confidence_threshold: 0.5      # dưới ngưỡng này: khung màu cam và nhãn có dấu "?"
detector:
  scale_factor: 1.1
  min_neighbors: 5
  min_face_size: 60
  detect_width: 640
```

## 4. Chức năng của ứng dụng

- Đọc luồng video IP camera bằng OpenCV, trong **thread riêng** để luôn lấy khung hình mới nhất (giảm độ trễ).
- Phát hiện khuôn mặt bằng Haar Cascade, dự đoán cảm xúc cho **nhiều khuôn mặt** trong cùng khung hình (các mặt được gom thành một batch).
- Hiển thị bounding box, nhãn cảm xúc, confidence, FPS và số khuôn mặt.
- Chọn mô hình bằng `--model`; chạy được trên **camera, ảnh tĩnh, file video**.
- Mất kết nối camera: in thông báo rõ ràng, hiện màn hình "Camera disconnected - reconnecting..." và **tự kết nối lại**, không crash.
- Tiền xử lý vùng mặt giống lúc huấn luyện (xem mục 6).

## 5. Dữ liệu

- Bộ dữ liệu: **FER-2013** (Kaggle: `msambare/fer2013`), ảnh xám 48×48, 7 lớp. Không dùng thêm dữ liệu ngoài.
- Tổng số ảnh: 28.709 ảnh train và 7.178 ảnh test.

| Lớp | Train | Test |
|---|---:|---:|
| angry | 3.995 | 958 |
| disgust | 436 | 111 |
| fear | 4.097 | 1.024 |
| happy | 7.215 | 1.774 |
| neutral | 4.965 | 1.233 |
| sad | 4.830 | 1.247 |
| surprise | 3.171 | 831 |

- Tập train được tách **90% train / 10% validation** (25.838 / 2.871 ảnh), có giữ tỉ lệ lớp (stratified, seed 42). Tập test dùng nguyên bản để đánh giá cuối.
- Xử lý mất cân bằng lớp: **class weight** (lấy căn bậc hai của tỉ lệ nghịch số ảnh để tránh trọng số lớp `disgust` quá lớn) kết hợp **augmentation có chủ đích**.

## 6. Mô hình và huấn luyện

Chi tiết trong `notebooks/train.ipynb`.

| | M1 – CNN tự thiết kế | M2 – ResNet18 (transfer learning) |
|---|---|---|
| Kiến trúc | 4 khối (Conv–BN–ReLU ×2 – MaxPool – Dropout) + 2 lớp Linear | ResNet18 pretrained ImageNet, thay lớp cuối (Dropout 0.5 + Linear) |
| Đầu vào | ảnh xám 1×48×48, chuẩn hoá (x−0.5)/0.5 | ảnh xám lặp 3 kênh, 96×96, chuẩn hoá theo ImageNet |
| Đóng băng | không | `conv1`, `bn1`, `layer1` |
| Augmentation | lật ngang, xoay ±10°, dịch ±10% | RandomResizedCrop (80–100%), lật ngang, xoay ±15°, ColorJitter, RandomErasing |
| Tối ưu | AdamW, lr 1e-3, weight decay 1e-4 | AdamW, lr 3e-4, weight decay 0,05, label smoothing 0,1 |
| Lịch học | CosineAnnealing, 40 epoch | CosineAnnealing, 12 epoch |
| Batch size | 64 | 64 |

Lấy trọng số epoch có accuracy validation cao nhất.

**Tiền xử lý khi chạy realtime** (`src/predict.py`) dùng đúng các phép biến đổi của tập val/test: `Grayscale → Resize → ToTensor → Normalize`, cùng kích thước và thống kê chuẩn hoá như bảng trên.

## 7. Kết quả chính (tập test FER-2013, 7.178 ảnh)

| Mô hình | Accuracy | Macro F1 | Số tham số | FPS suy luận (CPU) | Yêu cầu tối thiểu |
|---|---:|---:|---:|---:|---|
| M1 – CNN tự thiết kế | **66,47%** | 0,6337 | 1.765.991 | 220,2 | ≥ 60% – đạt |
| M2 – ResNet18 | **67,57%** | 0,6586 | 11.180.103 | 39,0 | ≥ 65% – đạt |

FPS suy luận là tốc độ riêng của mô hình (1 ảnh/lần, trên CPU), chưa gồm bước dò mặt và hiển thị.

FPS của cả ứng dụng khi chạy với camera (số FPS hiển thị trên khung hình): **khoảng 20 – 35 FPS** (dao động theo điều kiện chạy như độ phân giải camera, số khuôn mặt trong khung hình và tải của máy) – **đạt** yêu cầu ≥ 10 FPS.

F1-score theo từng lớp:

| Lớp | M1 | M2 |
|---|---:|---:|
| angry | 0,578 | 0,586 |
| disgust | 0,578 | 0,657 |
| fear | 0,455 | 0,507 |
| happy | 0,873 | 0,873 |
| neutral | 0,630 | 0,633 |
| sad | 0,546 | 0,567 |
| surprise | 0,777 | 0,787 |

Nhận xét ngắn: `happy` và `surprise` dễ nhận nhất; `fear`, `sad`, `angry` hay nhầm lẫn với nhau và với `neutral`. Phân tích lỗi chi tiết (confusion matrix, ảnh dự đoán sai) nằm trong `train.ipynb` và `report.pdf`.

## 8. Hạn chế đã biết

- **Haar Cascade** nhanh, chạy tốt trên CPU nhưng chỉ ổn định với mặt nhìn thẳng. Mặt nghiêng nhiều, cúi, bị che hoặc thiếu sáng có thể không dò được.
- Ảnh rất nhỏ (như 48×48 của FER-2013) được phóng to trước khi dò mặt; ảnh cận mặt (mặt sát mép) được thêm viền để dò lại.
- Mô hình huấn luyện trên ảnh mặt đã cắt sát và chính diện nên kém chính xác hơn với góc mặt lạ hoặc ánh sáng yếu.
- Dự đoán có confidence thấp hơn ngưỡng vẫn hiển thị nhưng có màu cam và dấu `?`.

## 9. Xử lý sự cố thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `Failed to resolve hostname <ip>` | Chưa thay `<ip>` bằng địa chỉ thật của điện thoại |
| Không kết nối được camera | Kiểm tra cùng Wi-Fi, mở thử địa chỉ camera trên trình duyệt laptop, kiểm tra cổng (`8080` hoặc `4747`) |
| `Không tìm thấy file trọng số` | Đặt `m1_cnn.pth` và `m2_resnet18.pth` vào thư mục `weights/` ở thư mục gốc dự án |
| `module 'cv2' has no attribute 'CascadeClassifier'` | Cài lại `pip install "opencv-python<5"` |
| `OMP: Error #15` | Đặt biến môi trường `KMP_DUPLICATE_LIB_OK=TRUE` |
| FPS thấp | Giảm độ phân giải trong ứng dụng camera xuống 640×480, dùng `--model cnn`, tăng `min_neighbors` hoặc `min_face_size` trong `config.yaml` |

## 10. Huấn luyện lại

Mở `notebooks/train.ipynb` và chạy tuần tự. Notebook tải dataset bằng `kagglehub`, cần thông tin đăng nhập Kaggle (đặt `KAGGLE_USERNAME` và `KAGGLE_KEY` trong biến môi trường hoặc file `~/.kaggle/kaggle.json`, **không ghi cứng khoá vào notebook**). Trọng số được lưu vào `weights/` (notebook nằm trong `notebooks/` nên lưu vào `../weights`).
