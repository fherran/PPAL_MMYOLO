## 📌 Description

**PPAL_MMYOLO** is an unofficial implementation of **Plug-and-Play Active Learning (PPAL)** for object detection, built on top of **MMYOLO** to leverage **YOLO-type models**.

⚠️ **Note:** This project is intended for **research and experimental use only** and is **not officially affiliated** with the original PPAL or MMYOLO authors.

## ⚙️ Installation

### 📦 Prerequisites (Recommended)

The following versions have been tested and are recommended for reproducibility:

- 🐍 **Python** == 3.8  
- 🔥 **PyTorch** == 1.10.1  
- 🧩 **MMCV** == 2.0.1  
- 🧠 **MMDetection (MMDet)** == 3.3.0  
- 🚀 **MMYOLO** == 0.6.0  
- ⚙️ **MMEngine** == 0.10.7  

---

### 🧪 Environment Setup

Create and activate a conda environment with the recommended PyTorch and CUDA versions:

```shell
conda create -n ppal_mmyolo python=3.8 pytorch==1.10.1 torchvision==0.11.2 cudatoolkit=11.3 -c pytorch -y
conda activate ppal_mmyolo

git clone https://github.com/fherran/PPAL_MMYOLO.git # Clone PPAL_MMYOLO
```

---

### 📚 Install OpenMMLab Core Dependencies
```shell
pip install openmim
mim install "mmcv==2.01"
mim install "mmengine==0.6.0"
mim install "mmdet==3.3.0"
```

---

### 🚀 Install MMYOLO
```shell
git clone https://github.com/open-mmlab/mmyolo.git
cd mmyolo

pip install -r requirements/albu.txt # For albumentations

mim install -v -e . # Install mmyolo
```

---

### + Extra Step
You need to install the depnedicies from this repo.

```shell
cd .. # If you still inside mmyolo

pip install -e .
```
## ✅ Quick Start / Usage

Currently, this repository provides **PPAL configurations for YOLOv7-Tiny only**.  

However, the implementation is **model-agnostic**, and the same approach can be extended to other YOLO variants by replicating and adapting the provided configuration files.

Please note that **additional effort may be required for different YOLO versions** (e.g., YOLOv8), as they may introduce changes in the **backbone**, **neck**, or **detection head**.

### Dataset Preparation

Before training, datasets with **YOLO-style annotations** must be converted to **COCO-style format**, which is required by MMYOLO.

You can perform this conversion using the official MMYOLO utility:

```shell
python mmyolo/tools/dataset_converters/yolo2coco.py
```

### Training with Active Learning
Before executing the active learning script, please follow the **original PPAL repository** to prepare the required **active learning dataset structure**. 

Additionally, ensure that all necessary settings in the configuration files are properly adjusted (e.g., number of GPUs, annotation budget, and active learning parameters).

Once the setup is complete, active learning can be launched using:
```shell
python tools/run_al_coco.py --config PATH_TO_PPAL_YOLO_CONFIG
```

---
## Contributions
We welcome contributions!  
If you implement a new model configuration (PPAL, MMDetection, or MMYOLO), feel free to open a pull request.

Thanks for your awesome work 🙂
