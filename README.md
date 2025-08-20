# The Loupe: A Plug-and-Play Attention Module for Vision Transformers

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Framework](https://img.shields.io/badge/Framework-PyTorch-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains the official PyTorch implementation for the research paper, **"The Loupe: A Plug-and-Play Attention Module for Amplifying Discriminative Features in Vision Transformers."**

The Loupe is a novel, lightweight, and intrinsically interpretable attention module designed to be seamlessly integrated into pre-trained Vision Transformer backbones like the Swin Transformer. It is trained end-to-end with a composite loss function that encourages the model to focus on small, discriminative regions, thereby improving performance and providing clear visual explanations for its decisions in Fine-Grained Visual Classification (FGVC) tasks.

---

## Key Results

Our primary contribution is a simple module that provides a significant performance gain over a strong baseline on the CUB-200-2011 dataset, while also offering valuable interpretability.

### Quantitative Improvement

The Swin-Loupe model, with a gentle sparsity penalty, achieves a **+2.66%** absolute accuracy improvement over an identically-trained baseline.

| Model                  | Accuracy (%) |
| ---------------------- | ------------ |
| Swin-Base (Our Baseline) | 85.40%       |
| **Swin-Loupe (Ours)** | **88.06%** |

### Qualitative Analysis: The Loupe in Action

The key feature of our module is its ability to learn, without any direct supervision, to focus on semantically meaningful and class-discriminative features. The visualizations below show the model's focus (green contours) on key regions like the head, eye, bill, and unique plumage patterns.

<img width="600" height="486" alt="image" src="https://github.com/user-attachments/assets/df5bffef-f3fd-476e-b8af-b1eae645ec70" />


*Fig. 1: Qualitative results from our best Swin-Loupe model. The module demonstrates a consistent ability to localize key discriminative features.*

---

## Project Structure

```
the-loupe/
│
├── data/                 # Organized CUB-200-2011 dataset will be created here
├── figures/              # For paper figures (architecture, visualizations)
├── output/               # Saved model weights (.pth files) will be stored here
├── src/                  # All Python source code
│   ├── model.py          # Defines the SwinWithLoupe (V1) architecture
│   ├── dataset.py        # Custom dataset for the masked loss experiment
│   ├── train.py          # Main training script for the V1 model
│   ├── train_masked.py   # Training script for the masked loss experiment
│   └── visualize_final.py  # Script to generate attention visualizations
│
└── README.md             # This file
```

---

## Setup and Installation

This project uses `uv` for fast and reliable package management.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/narensen/Loupe.git
    cd the-loupe
    ```

2.  **Create and activate the virtual environment:**
    ```bash
    uv venv
    source .venv/bin/activate
    ```

3.  **Install dependencies:**
    This command installs the correct CUDA-enabled version of PyTorch and all other required libraries.
    ```bash
    uv pip install --extra-index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121) torch torchvision
    uv pip install timm pandas tqdm opencv-python scikit-learn
    ```

---

## Usage

### 1. Prepare the Dataset

First, download and organize the CUB-200-2011 dataset. You can use the provided Colab notebook or a local script to download and structure the data into `data/train` and `data/test` directories.

### 2. Train the Model

You can train either the baseline model or our Swin-Loupe model. The training script uses the advanced training recipe we developed.

* **Train the Swin-Loupe (V1) Model:**
    This is the main experiment to reproduce our best result (88.06%).
    ```bash
    python src/train.py
    ```
    The best model weights will be saved in the `output/` directory.

* **Train the Baseline Model:**
    To reproduce the baseline result (85.40%), you will need to use a separate training script for a standard `timm` model.

### 3. Generate Visualizations

Once you have a trained model, you can generate the attention map visualizations using our final, professional script.

```bash
# Point to the weights of your best trained model
python visualize_final.py --weights ./output/best_model.pth --num_images 10
```

The output images will be saved in the `final_visualizations/` directory.

---

## Paper

For a complete technical description of the methodology and a full analysis of the results, please refer to our paper:

**[arxiv.org]**

### Citation

If you find our work useful in your research, please consider citing our paper:

```bibtex
@inproceedings{sengodan2025loupe,
  title={The Loupe: A Plug-and-Play Attention Module for Amplifying Discriminative Features in Vision Transformers},
  author={Sengodan, Naren and Gemini},
  booktitle={Proceedings of the IEEE INDICON},
  year={2025}
}
```

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
