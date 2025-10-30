# CRG-Diffusion: Controllable Region-Guided Diffusion for PCB Defect Generation

[![arXiv](https://img.shields.io/badge/arXiv-Paper-red)](https://arxiv.org)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

> **Official PyTorch Implementation**  
> **Status:** Under Review at IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)  
> **Contact:** xiaohang0608@foxmail.com

This repository contains the official implementation of our paper submitted to TCSVT, proposing a novel controllable region-guided diffusion model for PCB defect image generation with precise spatial control.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Method](#method)
- [Installation](#installation)
- [Pre-trained Models](#pre-trained-models)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
  - [Defect Generation](#defect-generation)
  - [Model Training](#model-training)
- [Configuration](#configuration)
- [Results](#results)
- [Citation](#citation)
- [Acknowledgments](#acknowledgments)
- [Contact](#contact)

---

## 🔍 Overview

PCB (Printed Circuit Board) defect detection is critical for quality control in electronic manufacturing. However, acquiring sufficient defect samples for training robust detection models remains challenging due to:

- **Scarcity of defect samples** in real-world production
- **Class imbalance** among different defect types
- **High annotation costs** for defect localization

Our CRG-Diffusion addresses these challenges by generating realistic PCB defect images with **precise spatial control** over defect locations and types, enabling effective data augmentation for defect detection models.

### Key Features

✅ **Precise Spatial Control**: Generate defects at specified locations with pixel-level accuracy  
✅ **Multi-Scale Generation**: Hierarchical diffusion process for both local (64×64) and global (256×256) synthesis  
✅ **Attention-Guided Refinement**: Dynamic attention mechanism for seamless defect-background integration  
✅ **High Fidelity**: Generates photorealistic PCB defect images indistinguishable from real samples  
✅ **Flexible Framework**: Supports 6 common PCB defect types with extensible architecture

---

## 🎯 Method

### Architecture Overview

<!-- You can add an architecture diagram here -->
<!-- ![Architecture](figures/architecture.png) -->

Our method employs a **two-stage hierarchical diffusion framework**:

#### Stage 1: Crop Region Generation (64×64)
- **Defect-Conditioned Diffusion**: Generates a 64×64 defect crop region conditioned on defect type and location
- **Local Detail Synthesis**: Focuses computational resources on the defect area for high-quality local features
- **Class-Conditional Guidance**: Ensures generated defects match the specified defect category

#### Stage 2: Global Image Composition (256×256)
- **Attention-Guided Inpainting**: Integrates the generated defect crop into the defect-free background
- **Dynamic Attention Mechanism**: Computes pixel-level attention masks to distinguish defect and background regions
- **Progressive Refinement**: Iteratively refines the composition through the reverse diffusion process

#### Key Components

1. **Dual-Model Architecture**: 
   - Crop model (64×64) for defect generation
   - Main model (256×256) for global composition

2. **Attention Mask Computation**:
   - Pixel-wise difference analysis between defect and defect-free crops
   - Temperature-controlled attention sharpness
   - Connected component filtering for robust mask extraction
   - Exponential moving average (EMA) for temporal consistency

3. **DDIM Sampling**:
   - Accelerated inference with fewer sampling steps
   - Deterministic sampling for reproducible generation

### Technical Highlights

- **Center-Weighted Radial Decay**: Prioritizes central defect regions while smoothly transitioning to background
- **Adaptive Thresholding**: Automatically adjusts attention masks based on defect characteristics
- **Attention Accumulation**: Maintains temporal coherence across diffusion timesteps

---

## � Project Structure

```
CRG-Diffusion/
├── checkpoint/                      # Pre-trained model checkpoints (download separately)
│   ├── checkpoint-64-defect.pt     # Crop region generator (64×64)
│   └── checkpoint-256-defect-free.pt # Main defect-free model (256×256)
├── confs/                          # Configuration files
│   ├── pcb_gen.yml                # Generation/inference config
│   ├── train-64.yml               # Training config for 64×64 model
│   └── train-256.yml              # Training config for 256×256 model
├── data/                           # Data directory
│   ├── pcb.json                   # Defect class definitions
│   └── sample/                    # Sample defect-free images and annotations
├── guided_diffusion/               # Core diffusion model implementation
│   ├── ddim.py                    # DDIM sampling and attention mechanism
│   ├── gaussian_diffusion.py     # Gaussian diffusion base class
│   ├── unet.py                    # U-Net architecture
│   └── ...
├── utils/                          # Utility modules
│   ├── config.py                  # Configuration management
│   ├── logger.py                  # Logging utilities
│   └── ...
├── sample_generation.py            # Main generation script
├── check_installation.py           # Installation verification script
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── QUICKSTART.md                   # Quick start guide
├── CONFIGURATION.md                # Detailed configuration reference
└── LICENSE                         # MIT License
```

---

## �🔧 Installation

### Prerequisites

- Python 3.8 or higher
- CUDA 11.0 or higher (for GPU acceleration)
- 8GB+ GPU memory recommended

### Step 1: Clone the Repository

```bash
git clone https://github.com/xiaohang0608/CRG-Diffusion.git
cd CRG-Diffusion
```

### Step 2: Create Virtual Environment

```bash
# Using conda (recommended)
conda create -n crg-diffusion python=3.8
conda activate crg-diffusion

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows
```

### Step 3: Install Dependencies

```bash
# Install PyTorch (CUDA 11.8 example)
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118

# Or for CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
# Run installation checker
python check_installation.py

# Or use the bash script
./check_installation.sh
```

<details>
<summary>Click to view <code>requirements.txt</code></summary>

```
numpy>=1.21.0
Pillow>=9.0.0
PyYAML>=6.0
scipy>=1.7.0
tqdm>=4.62.0
opencv-python>=4.5.0
scikit-image>=0.19.0
matplotlib>=3.5.0
tensorboard>=2.8.0
blobfile>=2.0.0
mpi4py>=3.1.0
```
</details>

---

## 📦 Pre-trained Models

We provide pre-trained models for both the crop region generator and global composition model.

### Download Links

| Model | Resolution | Parameters | Google Drive |
|-------|-----------|------------|--------------|
| Crop Model (Defect) | 64×64 | ~35M | [Download](https://drive.google.com/file/d/1K7-szMOXjQRb491VB-Niaq-BMFkj9XZl/view?usp=sharing) |
| Main Model (Defect-Free) | 256×256 | ~110M | [Download](https://drive.google.com/file/d/1YlQm2ByM_s6dl8wffuiN7fVpBGVfk9dI/view?usp=drive_link) |

### Setup Instructions

1. Download both checkpoint files from the links above
2. Create a `checkpoint` directory in the project root:
   ```bash
   mkdir checkpoint
   ```
3. Place the downloaded files with the following names:
   ```
   checkpoint/
   ├── checkpoint-64-defect.pt      # Crop model (64×64)
   └── checkpoint-256-defect-free.pt # Main model (256×256)
   ```

**Note**: The checkpoint files are ~500MB each. Ensure you have sufficient disk space and stable internet connection.

---

## 📊 Dataset Preparation

### PCB Defect Dataset

Our models are trained on the [PCB Defect Dataset](https://github.com/Charmve/Surface-Defect-Detection), which contains six defect types:

- **Missing Hole** (ID: 0)
- **Mouse Bite** (ID: 1)
- **Open Circuit** (ID: 2)
- **Short** (ID: 3)
- **Spur** (ID: 4)
- **Spurious Copper** (ID: 5)

### Data Organization

Organize your data as follows:

```
data/
├── pcb.json                    # Defect class definitions
└── sample/
    ├── defect_free_001.png     # Defect-free images
    ├── defect_free_001.txt     # Defect annotations (YOLO format)
    ├── defect_free_002.png
    ├── defect_free_002.txt
    └── ...
```

### Annotation Format

Defect annotations use YOLO format (`.txt` files):

```
<class_id> <x_center> <y_center> <width> <height>
```

Where coordinates are normalized to [0, 1]:
- `class_id`: Defect type ID (0-5)
- `x_center`, `y_center`: Center coordinates (relative to image size)
- `width`, `height`: Bounding box dimensions (relative to image size)

**Example** (`defect_free_001.txt`):
```
2 0.456 0.512 0.050 0.045
3 0.678 0.234 0.038 0.042
```

---

## 🚀 Usage

### Defect Generation

Generate PCB defect images using pre-trained models:

#### Basic Usage

```bash
python sample_generation.py --conf_path confs/pcb_gen.yml
```

#### Custom Configuration

```bash
python sample_generation.py \
    --conf_path confs/pcb_gen.yml \
    --seed 42 \
    --n_samples 8
```

#### Key Parameters (in `confs/pcb_gen.yml`)

```yaml
# Model paths
model_path: checkpoint/checkpoint-256-defect-free.pt
crop_model_path: checkpoint/checkpoint-64-defect.pt

# Input/Output
defect_free_dir: data/sample/          # Input defect-free images
outdir: ./images/                       # Output directory

# Generation settings
n_samples: 4                            # Number of samples per defect
n_iter: 1                               # Iterations per sample
seed: 42                                # Random seed (null for random)

# Attention mechanism
attention_features:
  temperature: 0.3                      # Attention sharpness (lower = sharper)
  center_weight: 1.0                    # Center region weight
  radial_decay: 0.15                    # Radial decay rate
  accumulation_beta: 0.9                # EMA coefficient

# DDIM sampling
ddim:
  num_inference_steps: 250              # Number of sampling steps
  ddim_sigma: 0.0                       # DDIM noise scale
```

#### Output Structure

```
images/
├── sample/                             # Generated defect images
│   ├── image_001_opencircuit_cx120_cy150_0.png
│   ├── image_001_opencircuit_cx120_cy150_1.png
│   └── ...
├── grid/                               # Grid visualization (multiple samples)
│   ├── image_001_opencircuit_cx120_cy150_grid.png
│   └── ...
└── log/                                # Generation logs
    └── generation.log
```

### Model Training

To train models from scratch, we recommend using the [Improved Diffusion](https://github.com/openai/improved-diffusion) codebase with our configuration files.

#### Training Crop Model (64×64)

```bash
python scripts/image_train.py \
    --data_dir /path/to/64x64/crops \
    --image_size 64 \
    --num_channels 128 \
    --num_res_blocks 2 \
    --class_cond True \
    --diffusion_steps 1000 \
    --noise_schedule linear \
    --lr 1e-4 \
    --batch_size 16
```

See `confs/train-64.yml` for detailed configuration.

#### Training Main Model (256×256)

```bash
python scripts/image_train.py \
    --data_dir /path/to/256x256/images \
    --image_size 256 \
    --num_channels 128 \
    --num_res_blocks 2 \
    --class_cond False \
    --diffusion_steps 1000 \
    --noise_schedule linear \
    --lr 1e-4 \
    --batch_size 8
```

See `confs/train-256.yml` for detailed configuration.

**Important Notes:**
- Ensure resolution settings match the target model size (64×64 or 256×256)
- The crop model requires class-conditional training (`class_cond=True`)
- Adjust `batch_size` based on available GPU memory
- Training typically requires 300K-500K iterations for convergence

---

## ⚙️ Configuration

### Attention Mechanism Parameters

Fine-tune the attention-guided composition in `confs/pcb_gen.yml`:

| Parameter | Range | Description | Default |
|-----------|-------|-------------|---------|
| `temperature` | 0.1-1.0 | Controls attention mask sharpness | 0.3 |
| `blur_sigma` | 0.5-2.0 | Gaussian blur for smooth boundaries | 1.0 |
| `center_weight` | 0.0-1.0 | Center region weighting strength | 1.0 |
| `radial_decay` | 0.0-1.0 | Radial decay from center to edges | 0.15 |
| `accumulation_beta` | 0.5-0.99 | EMA coefficient for attention | 0.9 |
| `min_area_ratio` | 0.001-0.1 | Min component area for filtering | 0.01 |
| `threshold_factor` | 0.3-0.8 | Adaptive threshold scaling | 0.6 |

### DDIM Sampling Configuration

Adjust inference speed vs. quality trade-off:

```yaml
ddim:
  num_inference_steps: 250    # Higher = better quality, slower
  ddim_sigma: 0.0             # 0.0 for deterministic sampling
```

Recommended steps:
- **Fast**: 50 steps (~5s per image)
- **Balanced**: 250 steps (~15s per image)
- **High Quality**: 1000 steps (~60s per image)

---

## 📈 Results

### Qualitative Results

<!-- Add sample results here -->
<!-- ![Results](figures/results.png) -->

Our method generates high-fidelity PCB defect images with:
- ✅ Realistic defect textures and patterns
- ✅ Precise spatial localization
- ✅ Seamless background integration
- ✅ Diverse defect appearances

### Quantitative Evaluation

| Metric | Value |
|--------|-------|
| FID ↓ | XX.XX |
| LPIPS ↓ | X.XXX |
| Detection mAP ↑ | XX.X% |

*Detailed experimental results will be updated upon paper acceptance.*

---

## 📝 Citation

If you find this work useful for your research, please consider citing:

```bibtex
@article{luo2025crg,
  title={CRG-Diffusion: Controllable Region-Guided Diffusion for PCB Defect Generation},
  author={Luo, Xiaohang and others},
  journal={IEEE Transactions on Circuits and Systems for Video Technology},
  year={2025},
  note={Under Review}
}
```

**Note:** The BibTeX entry will be updated with complete author information and DOI upon paper acceptance.

---

## 🙏 Acknowledgments

This work builds upon several excellent open-source projects:

- [Improved Diffusion](https://github.com/openai/improved-diffusion) - Training framework
- [DDIM](https://github.com/ermongroup/ddim) - Sampling algorithm
- [PCB Defect Dataset](https://github.com/Charmve/Surface-Defect-Detection) - Dataset

We thank the authors for their valuable contributions to the research community.

---

## 📧 Contact

For questions, suggestions, or collaboration opportunities, please contact:

- **Email**: xiaohang0608@foxmail.com
- **GitHub Issues**: [Report a bug or request a feature](https://github.com/xiaohang0608/CRG-Diffusion/issues)

**Useful Resources:**
- [Quick Start Guide](QUICKSTART.md) - Get started in minutes
- [Configuration Guide](CONFIGURATION.md) - Detailed parameter reference
- [FAQ](FAQ.md) - Frequently asked questions
- [Contributing Guidelines](CONTRIBUTING.md) - How to contribute

---

## 📄 License

This project is released under the [MIT License](LICENSE).

---

## 🔄 Updates

- **2025-10**: Repository initialized, pre-trained models released
- **2025-XX**: Paper submitted to IEEE TCSVT

---

<div align="center">

**⭐ If you find this project helpful, please consider giving it a star! ⭐**

</div>
