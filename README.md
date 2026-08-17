

<h1 align="center"> OmniCoT: A Benchmark for Global and Multi-Step Panoramic Reasoning </h1>

<div align="center">

[Haocong He](https://openreview.net/profile?id=%7EHaocong_He1)<sup>1*</sup>, [Chenfei Liao](https://chenfei-liao.github.io/)<sup>2*&dagger;</sup>, [Zichen Wen](https://scholar.google.com/citations?user=N-aPFvEAAAAJ&hl=zh-CN&oi=ao)<sup>1,7</sup>, [Zihao Dongfang](https://scholar.google.com/citations?user=IvJ4_xsAAAAJ&hl=zh-CN)<sup>2</sup>, [Xu Zheng](https://zhengxujosh.github.io/)<sup>2</sup>, [Bin Ren](https://amazingren.github.io/)<sup>3</sup>, [Chang Su](https://openreview.net/profile?id=~Chang_Su17)<sup>4</sup>, [Zixin Zhang](https://scholar.google.com/citations?user=BbZ0mwoAAAAJ&hl=zh-CN)<sup>2</sup>, [Harold H. Chen](https://haroldchen19.github.io/)<sup>2</sup>, [Hongfei Zhang](https://github.com/soyouthinkyoucantell)<sup>2</sup>, [Weijia Li](https://liweijia.github.io/)<sup>5</sup>, [Kailun Yang](https://yangkailun.com/)<sup>6</sup>, [Conghui He](https://conghui.ai/)<sup>7</sup>, [Xuming Hu](https://xuminghu.github.io/)<sup>2</sup>, [Nicu Sebe](https://disi.unitn.it/~sebe/)<sup>8</sup>, [Linfeng Zhang](https://www.zhanglinfeng.tech/)<sup>1&Dagger;</sup>

<sup>1</sup>SJTU, <sup>2</sup>HKUST(GZ), <sup>3</sup>MBZUAI, <sup>4</sup>JLU, <sup>5</sup>THU, <sup>6</sup>HNU, <sup>7</sup>Shanghai AI Lab, <sup>8</sup>UniTrento

<small>*Equal contribution &nbsp;&nbsp;&nbsp; &dagger;Project lead &nbsp;&nbsp;&nbsp; &Dagger;Corresponding author</small>

</div>

<div align="center">
    <a href="#"><img src="https://img.shields.io/badge/Project-Page-blue?style=for-the-badge&logo=github&logoColor=white" alt="Project Page"></a>
    <a href="#"><img src="https://img.shields.io/badge/Paper_(arXiv)-B31B1B?style=for-the-badge&logo=arxiv&logoColor=white" alt="Paper"></a>
    <br>
    <a href="https://huggingface.co/datasets/Eustia1/OmniCoT"><img src="https://img.shields.io/badge/Dataset-HuggingFace-orange?style=for-the-badge&logo=huggingface&logoColor=white" alt="Dataset"></a>
    <a href="https://huggingface.co/Eustia1/OmniCoT-R1"><img src="https://img.shields.io/badge/Model-OmniCoT--R1-orange?style=for-the-badge&logo=huggingface&logoColor=white" alt="Model"></a>
</div>
<br>

Official repository for the paper: **OmniCoT: A Benchmark for Global and Multi-Step Panoramic Reasoning**.

> Multimodal Large Language Models (MLLMs) have demonstrated promising spatial reasoning capabilities, while these abilities remain underexplored in the emerging visual modality of panoramic imagery. The full 360° × 180° field of view of panoramas essentially supports complex global multi-step reasoning, which is also the fundamental advantage of panoramas in applications such as embodied intelligence. In this paper, we introduce **OmniCoT**, a panoramic spatial reasoning suite designed to enable MLLMs to use global evidence and perform multi-step inference across viewpoints.

<br>

<div align="center">
    <img src="assets/img/Teaser.png" alt="teaser" width="90%">
</div>

## 🚀 News
* **[2026-06]** 📄 Code is made publically available.
* **[2026-06]** 📄 OmniCoT is accepted by ECCV 2026!
* **[2026-03]** 📄 Paper is under review.

---

## 🌟 Highlights
- **OmniCoT-B Benchmark (6.7K):** A new benchmark requiring MLLMs to fully use the 360° space and perform multi-hop reasoning, moving beyond simplistic queries that rely on local cues. It measures both answer accuracy and Chain-of-Thought (CoT) quality.
- **OmniCoT-T Training Set (14.3K):** A purpose-built training set with structured stepwise Chain-of-Thought annotations that explicitly link intermediate reasoning steps to panoramic evidence.
- **OmniCoT-Real (1K):** A manually annotated real-world subset to quantify the Sim-to-Real gap.
- **OmniCoT-R1 Model:** Model weights are available on Hugging Face at [Eustia1/OmniCoT-R1](https://huggingface.co/Eustia1/OmniCoT-R1).
- **OmniCoT-R1 Baseline:** A baseline model developed via a two-stage strategy: SFT to anchor reasoning to panoramic evidence, and GRPO to penalize geometrically incoherent paths, consolidating global 360° spatial consistency.

---


## Quick Start

This repository contains:

- `Omni-COT/`: the OmniCoT QA and CoT generation pipeline.
- `lmms-eval/`: a vendored `lmms-eval` copy with local OmniCoT task configs under `lmms_eval/tasks/omnicot/`.

### 1. Clone

```bash
git clone https://github.com/Chenfei-Liao/OmniCoT.git
cd OmniCoT
```

### 2. Install Omni-COT

```bash
cd Omni-COT
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure an OpenAI-compatible API

The default config reads credentials and model names from environment variables:

```bash
export OPENAI_API_KEY="your_api_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export OMNICOT_REASONING_MODEL="your_reasoning_model"
export OMNICOT_TEXT_MODEL="your_text_model"
export OMNICOT_VISION_MODEL="your_vision_model"
```

If you use one model for every stage, set all three `OMNICOT_*_MODEL` variables to the same model name.

### 4. Run Omni-COT generation

Prepare a dataset root where each scene is stored in its own folder and includes `scene_data.json`, then run:

```bash
python run.py \
  --data-root path/to/osr_scenes \
  --output data/outputs/simplified_batch_output.json \
  --config config/api_config.yaml \
  --batch-config config/batch_config.yaml \
  --question-batches 2 \
  --target-qa-per-type 1 \
  --max-workers 5
```

Runtime caches are stored under `data/cache/`, so rerunning the same command can resume from previous stages.

### 5. Export accepted QA pairs

```bash
python src/stage_cache.py \
  --cache-dir data/cache/stage_cache \
  --export data/outputs/exported_qa.json
```

### 6. Evaluate OmniCoT with lmms-eval

The repository includes a bundled smoke-test subset with six real OmniCoT cases
and the corresponding ERP panoramic image:

```text
lmms-eval/lmms_eval/tasks/omnicot/sample_data/OmniCoT_sample.json
lmms-eval/lmms_eval/tasks/omnicot/sample_data/image/
```

Install the local `lmms-eval` copy (requires Python 3.10 or later):

```bash
cd ../lmms-eval
python -m pip install --upgrade pip
pip install -e .
```

Run the bundled OmniCoT smoke test:

```bash
python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct \
  --tasks omnicot_no_desc \
  --batch_size 1 \
  --limit 8 \
  --output_path results/omnicot_smoke
```

Available local tasks:

- `omnicot_no_desc`: visual input plus question.
- `omnicot_with_desc`: visual input plus structured scene description plus question.
- `omnicot_text_only`: text-only ablation.
- `omnicot_no_thinking`: direct-answer ablation.

For full benchmark evaluation, place the complete OmniCoT JSON and image folder
in a reproducible data location, update
`lmms_eval/tasks/omnicot/_default_template.yaml`, and set `OMNICOT_DATA_DIR` /
`OMNICOT_IMAGE_DIR` if your image paths are stored outside the JSON directory.

The answer-evaluation run writes a model submission file to:

```text
lmms-eval/results/omnicot_smoke/submissions/omnicot_submission.json
```

### 7. Evaluate CoT quality

After answer evaluation, run the CoT-quality judge on the generated submission:

```bash
export OPENAI_API_KEY="your_judge_api_key"
export OPENAI_BASE_URL="https://api.openai.com/v1"
export JUDGE_MODEL="your_judge_model"

python tools/omnicot_cot_quality_eval.py \
  --input-file results/omnicot_smoke/submissions/omnicot_submission.json \
  --output-dir results/omnicot_smoke/cot_quality \
  --mode all \
  --num-threads 4
```

The script supports three modes:

- `simple`: CoT precision, recall, and F1 against the reference CoT.
- `spatial`: viewpoint consistency, spatial evidence sufficiency, and reasoning feasibility.
- `all`: runs both groups of metrics.

`simple` uses two judge calls per sample, `spatial` uses three, and `all` uses
five. Use `--max-samples` for a small-cost smoke test before full evaluation.

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

OmniCoT is released as an ImageFolder dataset with one `metadata.jsonl`
file per split:

```text
train/
  images/
  metadata.jsonl
validation/
  images/
  metadata.jsonl
test/
  images/
  metadata.jsonl
real/
  images/
  metadata.jsonl
```

Each line in `metadata.jsonl` is one QA sample. Multiple QA samples may
reference the same panoramic image through `file_name`.

Split statistics:

| Split | QA samples | Unique images |
| --- | ---: | ---: |
| `train` | 14,385 | 2,800 |
| `validation` | 3,060 | 600 |
| `test` | 3,115 | 600 |
| `real` | 1,073 | 200 referenced real-world images |

Images are disjoint across splits to avoid leakage.

Fields:

| Field | Description |
| --- | --- |
| `image` | Panoramic image generated by the ImageFolder loader. |
| `file_name` | Relative image path within the split folder, for example `images/xxx.png`. |
| `scene_id` | Scene identifier. |
| `qa_id` | Unique QA sample id. |
| `type` | Question type, for example `viewpoint_transform_identify`. |
| `subtype` | Subtype label, for example `A1`. |
| `question` | Question text. |
| `answer` | Answer text. |
| `cot` | Chain-of-thought steps as a list of strings; may be empty for real-world samples. |
| `random_objects` | Optional list of objects used for randomization. |

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
