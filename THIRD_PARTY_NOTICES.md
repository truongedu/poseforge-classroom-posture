# Third-party software and model notices

PoseForge's repository license applies to original project source only. It does **not** relicense external packages, pretrained weights, or user recordings. No third-party model binaries or private datasets are shipped in the public source repository.

| Component | Use in PoseForge | Upstream license / terms |
|---|---|---|
| [Ultralytics YOLO11 / Ultralytics Python](https://github.com/ultralytics/ultralytics) | Pretrained YOLO11-Pose detector and ByteTrack integration | [AGPL-3.0 or Enterprise](https://docs.ultralytics.com/), subject to upstream conditions |
| [PySide6 / Qt for Python](https://doc.qt.io/qtforpython-6/) | Collector and inference desktop UI | LGPLv3/GPLv3 or Qt commercial; some Qt components have additional terms |
| [PyTorch](https://github.com/pytorch/pytorch) | Training and inference of project classifiers | See upstream license and notices |
| [OpenCV](https://github.com/opencv/opencv) | Video/image capture and overlays | See upstream license and notices |
| [NumPy](https://github.com/numpy/numpy), [pandas](https://github.com/pandas-dev/pandas), [scikit-learn](https://github.com/scikit-learn/scikit-learn), [Matplotlib](https://github.com/matplotlib/matplotlib) | Data engineering and evaluation | See individual upstream licenses |

The complete third-party obligations are determined by the exact versions and distributions installed by the user. If redistributing binaries, weights, or a bundled installer, review and include all licenses and notices required by those distributions, including Qt's applicable obligations.

**Data privacy:** Original subject recordings, consent documents, camera URLs, credentials, and identifiable personal metadata are not licensed or published by this repository. Only publish demonstration screenshots after checking the consent and privacy of all subjects shown.
