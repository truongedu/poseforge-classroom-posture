# PoseForge — RUN (Ubuntu, Conda `lab2_imp`)

**Chạy từ project root.** Các lệnh bên dưới độc lập; thay path nếu dữ liệu/model nằm nơi khác.

```bash
cd ~/truong_CODE/lab2_imp
conda activate lab2_imp
```

**Mặc định vận hành:** 1 nguồn, YOLO11n-Pose **GPU 0**, `MLP Wide` **GPU 0**, PT FP32. GRU có thể chọn trong Advanced Settings. Classifier CPU chỉ dùng khi chủ động so benchmark hoặc không có CUDA. Dataset V2 ở `artifacts/dataset_working/Real_Posture_Dataset`; **không sửa** `artifacts/dataset_backup/`.

## 1. Inference — giao diện (dùng hằng ngày)

```bash
python -m inference.gui --device 0 --cls-device cuda:0
```

- **Source:** `camera` + `0` (USB), URL `rtsp://...` hoặc `http://...` (IP, đúng endpoint stream), hoặc `video`/`image` + đường dẫn file. **Preview Sources** để xem góc trước khi START; **Show pose in Preview** chỉ chạy YOLO (không MLP).
- **START / END:** `Testcase` chạy liên tục trong một case; END reset, START mở case mới. `Continuous`: END dừng session. `Fast` hiển thị dự đoán ngay; `Guarded` đợi nhãn ổn định.
- **ROI:** vẽ ≥3 điểm để lọc người *sau YOLO*; không crop đầu vào. **Advanced Settings:** chọn YOLO weights `.pt` và thư mục checkpoint MLP/GRU (`model_final.pt`), ngưỡng confidence, stable frames, render FPS; chỉ chỉnh khi idle.
- **FPS:** YOLO Model, classifier (MLP/GRU), Pipeline và Live đo riêng. Render FPS không giới hạn throughput inference.

**Đổi thiết bị:** `--device 0|cpu` chỉ áp dụng YOLO; `--cls-device cuda:0|cpu` chỉ áp dụng classifier. Để so MLP CPU mà giữ YOLO GPU: `--device 0 --cls-device cpu`.

### Inference CLI — không mở PySide6

```bash
# Ảnh/video; kết quả tạo trong artifacts/inference/
python -m inference.run --source /path/to/video.mp4 --mode video \
  --pose artifacts/pretrained/yolo11n-pose.pt \
  --model artifacts/models/mlp_wide/v002 \
  --device 0 --cls-device cuda:0 --backend pt --headless

# Camera 0 hoặc RTSP: thay 0 bằng URL; Ctrl+C để dừng
python -m inference.run --source 0 --mode camera \
  --device 0 --cls-device cuda:0 --backend pt --headless
```

`--source` là video/image path, camera index hoặc URL; `--overlay` bật skeleton trên output; `--record-events` ghi clip khi class ổn định (camera). `--stable-frames 5` = số dự đoán liên tiếp cần ổn định; `--min-conf 0.55` = ngưỡng confidence. `--max-frames N` giới hạn test video; `--max-seconds N` giới hạn camera. Mặc định không ghi clip.

## 2. Benchmark FPS pipeline (headless)

```bash
python -B scripts/benchmark_inference_pipeline.py \
  --data artifacts/dataset_working/Real_Posture_Dataset \
  --pose artifacts/pretrained/yolo11n-pose.pt \
  --classifier artifacts/models/mlp_wide/v002 \
  --cls-backend pt --device 0 --cls-device cuda:0 \
  --frames 720 --warmup 20 --rounds 3 --seed 302 --overlay on
```

**Đo GRU trên cùng workload:** giữ nguyên lệnh, chỉ đổi `--classifier artifacts/models/gru_temporal/v001`. CSV vẫn có khóa legacy `mlp_forward_ms` cho cả hai họ; xem `classifier_family` để đọc đúng.

- **So classifier CPU:** chỉ đổi thành `--cls-device cpu`; YOLO vẫn GPU. Nên đo xen kẽ CPU/GPU và giữ nguồn, seed, profile điện năng giống nhau.
- `--frames`: frame/lượt; `--rounds`: số lượt; `--warmup`: frame làm nóng không tính FPS; `--seed`: lấy mẫu có thể tái lập; `--overlay on|off`: cùng điều kiện khi A/B. Có thể dùng `--video /path/file.mp4` **thay** `--data`.
- Output tự tăng version: `artifacts/benchmarks/inference_pipeline/vNNN/` (`summary.csv`, `runs.csv`, `stage_latency.csv`, `config.json`, `report.png`). **YOLO/classifier Model FPS** là forward throughput; **Pipeline FPS** không gồm decode/UI; **Live FPS** phải đo khi chạy nguồn thực. Các stage latency có quan hệ lồng nhau, **không cộng lại**.
- FPS phụ thuộc GPU clock/power và CPU scheduling. FPS tăng khi MLP chạy GPU không chứng minh toàn bộ mức tăng chỉ do MLP.

## 3. Benchmark giữa các YOLO Pose

```bash
# Tải n/s PT và tạo n/s ONNX (chỉ cần làm một lần; cần onnx)
python -B scripts/prepare_yolo_pose.py --variants n s \
  --export-onnx --export-device 0

# So sánh n/s × PT/ONNX trên cùng bộ frame
python -B scripts/benchmark_yolo_pose.py \
  --data artifacts/dataset_working/Real_Posture_Dataset \
  --pretrained artifacts/pretrained --device 0 \
  --frames 96 --warmup 3 --seed 302
```

Cần đủ bốn file `yolo11{n,s}-pose.{pt,onnx}`; script này **không** benchmark YOLO11m. Output: `artifacts/benchmarks/pose_n_vs_s_vNNN/comparison.csv` và `results.json`. Sai khác keypoint/selection là **độ đồng thuận tương đối**, không phải độ chính xác so với ground truth.

Đo tải YOLO kéo dài (khi cần):

```bash
python -B scripts/benchmark_sustained_pose.py --models pt \
  --data artifacts/dataset_working/Real_Posture_Dataset \
  --pretrained artifacts/pretrained --classifier artifacts/models/mlp_wide/v002 \
  --device 0 --frames 720 --rounds 4 --warmup 8
```

## 4. Model: tải, cài final checkpoint, xuất ONNX

```bash
# Tải checkpoint YOLO11n/s Pose (.pt), không ghi đè file sẵn có
python -B scripts/prepare_yolo_pose.py --variants n s

# Cài final_model_bundle*.zip đã train trên Colab (tạo version vNNN mới)
bash scripts/install_trained_model.sh ~/Downloads/final_model_bundle.zip

# Xuất MLP final v002 sang ONNX (PT vẫn giữ nguyên; kiểm tra parity)
python -B scripts/export_onnx.py --model artifacts/models/mlp_wide/v002

# Nếu muốn xuất YOLO Pose ONNX riêng
python -B scripts/export_onnx.py --yolo artifacts/pretrained/yolo11n-pose.pt \
  --imgsz 640 --device 0
```

`--model` là **thư mục** checkpoint có `model_final.pt`, không phải đường dẫn PT. Export chỉ hỗ trợ kiến trúc đã đăng ký; từ chối ghi đè. Inference chính thức vẫn dùng `--backend pt`; ONNX là tùy chọn test (`--backend onnx`, provider phụ thuộc package). Không train lại `mlp_wide/v002` chỉ vì export.

## 5. Collector và dataset V2

```bash
# Thu dữ liệu mới (mặc định runtime riêng, không chạm working/backup)
bash scripts/run_collector.sh --data artifacts/collector_runtime/Real_Posture_Dataset

# Mở một dataset cụ thể để kiểm tra/curate
bash scripts/run_collector.sh --data artifacts/dataset_working/Real_Posture_Dataset

# Kiểm tra raw (read-only)
python -B scripts/preflight_dataset.py \
  --data artifacts/dataset_working/Real_Posture_Dataset --decode-videos
python -B scripts/validate_dataset.py \
  --data artifacts/dataset_working/Real_Posture_Dataset --deep
```

Collector: **Cameras → Save & Connect**, chọn person/session/label, **S** bắt đầu take, **E** kết thúc; bật *Enable pose* nếu cần preview keypoints. Dataset V2 có curation/replay/split trong UI; không dùng Track ID làm khóa dữ liệu.

### Re-extract keypoints (chỉ khi cần)

```bash
# Dry-run: xem clip thiếu YOLO17 trước, không sửa dataset
python -B scripts/reextract_pose.py \
  --data artifacts/dataset_working/Real_Posture_Dataset \
  --model artifacts/pretrained/yolo11n-pose.pt --mode missing

# Thực sự re-extract clip thiếu (có backup/rollback theo công cụ)
python -B scripts/reextract_pose.py \
  --data artifacts/dataset_working/Real_Posture_Dataset \
  --model artifacts/pretrained/yolo11n-pose.pt --mode missing --apply
```

`--mode selected --person TQB --label 2 --apply --confirm-overwrite` dùng để xử lý lại subset có chủ đích; **không chạy trên backup**. Dùng nhất quán checkpoint YOLO với nguồn keypoint của dataset.

### Export ZIP sạch → notebook V2

```bash
# Materialize curation, không sửa working dataset
python -B scripts/export_clean_dataset.py \
  --data artifacts/dataset_working/Real_Posture_Dataset

# Kiểm tra ZIP sạch (tùy chọn)
python -B scripts/audit_clean_zip.py \
  artifacts/dataset_exports/clean_raw_v001.zip
```

Output là `artifacts/dataset_exports/clean_raw_vNNN.zip` (chọn **version mới nhất**, không mặc định v001). Upload ZIP lên Drive/Colab, cập nhật `POSEFORGE_ZIP` hoặc `RAW_ZIP` trong notebook; chạy theo thứ tự **`01_raw_eda_split_v2` → `02_process_eda_export_v2` → `03_train_mlp_v2` → `04_train_gru_v2`**. Notebook 03 cần cả raw ZIP để xác minh provenance và processed ZIP; Notebook 04 chỉ cần processed ZIP. Notebook 03 chỉ chạy nếu **chủ động train**, không phải bước inference hằng ngày.

## 6. Tests và tra tham số

```bash
bash scripts/test.sh quick  # unit/contract chính
bash scripts/test.sh full   # non-UI regression
bash scripts/test.sh ui     # collector Qt, offscreen

# Inference GUI Qt smoke (tránh pytest plugin ROS Humble)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH="$PWD:$PWD/collector" \
  QT_QPA_PLATFORM=offscreen python -B -m pytest -q tests/test_phase33_gui.py
```

Cần một biến CLI khác? Xem trực tiếp `python -B <script>.py --help` hoặc `python -m inference.run --help`. **Không dùng** `test_v3.sh`, notebook V1 hoặc script `retire_legacy.py` (đã xóa).
