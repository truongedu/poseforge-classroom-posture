# PoseForge Collector — Ubuntu / Conda

Collector dùng để **ghi raw video/keypoints, curate, replay và export dataset V2**; **không phải** inference UI Phase 3.3.

```bash
cd ~/truong_CODE/lab2_imp
conda activate lab2_imp
bash scripts/run_collector.sh --data artifacts/collector_runtime/Real_Posture_Dataset
# Muốn mở working dataset có sẵn:
bash scripts/run_collector.sh --data artifacts/dataset_working/Real_Posture_Dataset
```

Ghi vào `artifacts/collector_runtime/Real_Posture_Dataset/` (đặt `--data` tường minh để tránh guard dành cho thư mục V1 cũ). Backup tại `artifacts/dataset_backup/` **không chỉnh sửa**. Chọn camera qua **Save & Connect**, person/session/label, **S** bắt đầu take / **E** kết thúc; *Enable pose* để xem skeleton.

| Nhiệm vụ | Lệnh / nơi xem |
|---|---|
| Validation (read-only) | `python -B scripts/validate_dataset.py --data artifacts/dataset_working/Real_Posture_Dataset --deep` |
| Preflight MP4 | `python -B scripts/preflight_dataset.py --data artifacts/dataset_working/Real_Posture_Dataset --decode-videos` |
| Replay / curation / split | Tab trong Collector |
| Export clean raw ZIP | `python -B scripts/export_clean_dataset.py --data artifacts/dataset_working/Real_Posture_Dataset` |
| Collector UI tests | `bash scripts/test.sh ui` |

**[docs/RUN.md](../docs/RUN.md)** là nơi tra đầy đủ lệnh chạy inference, benchmark, models, export dataset và biến cần chỉnh. Các công cụ verify/re-extract vẫn ở `scripts/`; không dùng notebook V1/legacy scripts đã xóa.
