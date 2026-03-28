<h1 align="center"> OmniCoT: A Benchmark for Global and Multi-Step Panoramic Reasoning </h1>

<div align="center">

[Haocong He]()<sup>1*</sup>, [Chenfei Liao](https://chenfei-liao.github.io/)<sup>2*&dagger;</sup>, [Zichen Wen](https://scholar.google.com/citations?user=N-aPFvEAAAAJ&hl=zh-CN&oi=ao)<sup>1,7</sup>, [Zihao Dongfang]()<sup>2</sup>, [Xu Zheng](https://zhengxujosh.github.io/)<sup>2</sup>, [Bin Ren](https://amazingren.github.io/)<sup>3</sup>, [Chang Su]()<sup>4</sup>, [Zixin Zhang](https://scholar.google.com/citations?user=BbZ0mwoAAAAJ&hl=zh-CN)<sup>2</sup>, [Harold H. Chen](https://haroldchen19.github.io/)<sup>2</sup>, [Hongfei Zhang](https://github.com/soyouthinkyoucantell)<sup>2</sup>, [Weijia Li]()<sup>5</sup>, [Kailun Yang]()<sup>6</sup>, [Conghui He]()<sup>7</sup>, [Xuming Hu](https://xuminghu.github.io/)<sup>2</sup>, [Nicu Sebe](https://disi.unitn.it/~sebe/)<sup>8</sup>, [Linfeng Zhang]()<sup>1&Dagger;</sup>

<sup>1</sup>SJTU, <sup>2</sup>HKUST(GZ), <sup>3</sup>MBZUAI, <sup>4</sup>JLU, <sup>5</sup>THU, <sup>6</sup>HNU, <sup>7</sup>Shanghai AI Lab, <sup>8</sup>UniTrento

<small>*Equal contribution &nbsp;&nbsp;&nbsp; &dagger;Project lead &nbsp;&nbsp;&nbsp; &Dagger;Corresponding author</small>

</div>

<div align="center">
    <a href="#"><img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=github&logoColor=white" alt="Project Page"></a>
    <a href="#"><img src="https://img.shields.io/badge/Paper_(arXiv)-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"></a>
    <br>
    <a href="https://huggingface.co/datasets/Eustia1/OmniCoT"><img src="https://img.shields.io/badge/Dataset-HuggingFace-orange?style=for-the-badge&logo=huggingface&logoColor=white" alt="Dataset"></a>
</div>
<br>

Official repository for the paper: **OmniCoT: A Benchmark for Global and Multi-Step Panoramic Reasoning**.

> Multimodal Large Language Models (MLLMs) have demonstrated promising spatial reasoning capabilities, while these abilities remain underexplored in the emerging visual modality of panoramic imagery. The full 360° × 180° field of view of panoramas essentially supports complex global multi-step reasoning, which is also the fundamental advantage of panoramas in applications such as embodied intelligence. In this paper, we introduce **OmniCoT**, a panoramic spatial reasoning suite designed to enable MLLMs to use global evidence and perform multi-step inference across viewpoints.

<br>

<div align="center">
    <img src="assets/img/Teaser.png" alt="teaser" width="90%">
</div>

## 🚀 News
* **[2026-03]** 🔥 [OmniCoT Dataset](#-omnicot-dataset) and OmniCoT-R1 model will be released soon!
* **[2026-03]** 📄 Paper is under review.

---

## 🌟 Highlights
- **OmniCoT-B Benchmark (6.7K):** A new benchmark requiring MLLMs to fully use the 360° space and perform multi-hop reasoning, moving beyond simplistic queries that rely on local cues. It measures both answer accuracy and Chain-of-Thought (CoT) quality.
- **OmniCoT-T Training Set (14.3K):** A purpose-built training set with structured stepwise Chain-of-Thought annotations that explicitly link intermediate reasoning steps to panoramic evidence.
- **OmniCoT-Real (1K):** A manually annotated real-world subset to quantify the Sim-to-Real gap.
- **OmniCoT-R1 Baseline:** A baseline model developed via a two-stage strategy: SFT to anchor reasoning to panoramic evidence, and GRPO to penalize geometrically incoherent paths, consolidating global 360° spatial consistency.

---

## 🛠️ Environment Setup
### 1. Install Dependencies
```bash
conda create -n omnicot python=3.10
conda activate omnicot
pip install -r requirements.txt
```
*(Detailed environment setup instructions will be updated upon code release.)*

---

## 📊 OmniCoT Dataset

### Downloading the Dataset
You can download the dataset directly from our Hugging Face repository:

#### [Hugging Face](https://huggingface.co/datasets/Eustia1/OmniCoT)
```bash
huggingface-cli download --repo-type dataset Eustia1/OmniCoT --local-dir /path/to/OmniCoT
```

### Dataset Structure
The dataset follows the "See-Locate-Move" taxonomy, evaluating multi-hop viewpoint transformation, inter-object spatial relationship, and interactive multi-step planning. 

*(Detailed structure will be updated soon.)*

---

## 🚀 Quick Demo
*(Inference and evaluation code will be provided soon.)*

---

## 💬 Citation
If you find our work helpful, please cite:
```bibtex
@article{omnicot2026,
  title   = {OmniCoT: A Benchmark for Global and Multi-Step Panoramic Reasoning},
  author  = {He, Haocong and Liao, Chenfei and Wen, Zichen and Dongfang, Zihao and Zheng, Xu and Ren, Bin and Su, Chang and Zhang, Zixin and Chen, Harold Haodong and Zhang, Hongfei and Li, Weijia and Yang, Kailun and He, Conghui and Hu, Xuming and Sebe, Nicu and Zhang, Linfeng},
  journal = {arXiv preprint},
  year    = {2026}
}
```

---

## 📧 Contact
If you have any questions or suggestions, please feel free to contact us at [cliao127@connect.hkust-gz.edu.cn](mailto:cliao127@connect.hkust-gz.edu.cn).
