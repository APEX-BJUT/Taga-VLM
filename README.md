# TagaVLM: Topology-Aware Global Action Reasoning for Vision-Language Navigation

<p align="center">
  <a href="https://apex-bjut.github.io/Taga-VLM/"><img src="https://img.shields.io/badge/Project-Page-blue" alt="Project Page"></a>
  <a href="https://huggingface.co/tiredtony"><img src="https://img.shields.io/badge/%F0%9F%A4%97-Model%20Weights-yellow" alt="HuggingFace"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green" alt="License"></a>
  <img src="https://img.shields.io/badge/ICRA-2026-red" alt="ICRA 2026">
</p>

Official implementation of the ICRA 2026 paper **"TagaVLM: Topology-Aware Global Action Reasoning for Vision-Language Navigation"**.

For details, please visit our [project page](https://apex-bjut.github.io/Taga-VLM/).

![TagaVLM Framework](assets/framework.png)

## Results on R2R (Val Unseen)

| Method | Backbone | NE ↓ | OSR ↑ | SR ↑ | SPL ↑ |
|--------|----------|------|-------|------|-------|
| NavCoT | LLaMA2-7B | 6.26 | 48.11 | 40.23 | 36.64 |
| MapGPT | GPT-4V | 5.62 | 57.9 | 47.7 | 38.1 |
| **TagaVLM-0.5B (Ours)** | Qwen2-0.5B | 5.57 | 55.09 | 45.72 | 41.91 |
| **TagaVLM-7B (Ours)** | Qwen2-7B | **4.97** | **60.2** | **51.09** | **47.18** |

## Installation

```bash
git clone https://github.com/APEX-BJUT/Taga-VLM.git
cd Taga-VLM

conda create -n tagavlm python=3.9 -y
conda activate tagavlm
pip install --upgrade pip
pip install -e ".[train]"
```

Install the patched transformers (required for STAR-Att):

```bash
cd transformers-4.40.0 && pip install -e . && cd ..
```

Additional pinned dependencies: `accelerate==0.28.0`, `numpy<=2.0`.

**Flash-Attention 2:** Download the prebuilt `.whl` for your CUDA/Python version from [Flash-Attention Releases](https://github.com/Dao-AILab/flash-attention/releases) (select the `abiFALSE` variant), then:

```bash
pip install flash_attn-*.whl
```

**Matterport3D Simulator:** Follow [Matterport3DSimulator](https://github.com/peteanderson80/Matterport3DSimulator).

## Data Preparation

Download model weights and data from [HuggingFace](https://huggingface.co/tiredtony) and place them as:

```
Taga-VLM/
├── data/
│   ├── mp3d_pano_images/
│   ├── mp3d_views/
│   └── anno/
└── model_zoo/
    ├── TagaVLM-7b/
    └── TagaVLM-0.5b/
```

## Training & Evaluation

```bash
# Training
bash scripts/train/finetune_ov_test.sh

# Evaluation on R2R
cd map_nav_src && bash run_r2r.sh
```

## Citation

```bibtex
@inproceedings{liu2026tagavlm,
  title     = {TagaVLM: Topology-Aware Global Action Reasoning for Vision-Language Navigation},
  author    = {Liu, Jiaxing and Zhang, Zexi and Li, Xiaoyan and Wang, Boyue and Hu, Yongli and Yin, Baocai},
  booktitle = {Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)},
  year      = {2026}
}
```

## Acknowledgement

This project builds upon [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT) and [VLN-DUET](https://github.com/cshizhe/VLN-DUET). We thank the authors for open-sourcing their code.
