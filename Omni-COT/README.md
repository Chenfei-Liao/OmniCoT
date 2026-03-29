# Omni-COT

Omni-COT is a batch pipeline for generating spatial reasoning QA pairs from scene-level structured data. The open-source release focuses on a prompt-driven generation workflow built on top of an OpenAI-compatible API stack.

Following the paper, OmniCoT is organized around three progressive dimensions:

- `See`: multi-hop viewpoint transformation
- `Locate`: inter-object spatial relationship reasoning
- `Move`: embodied action simulation

The current release implements 6 question types:

- `Type A1 / MOT / viewpoint_transform_identify`
- `Type A2 / RAC / viewpoint_transform_angle`
- `Type B1 / MOI / multi_hop_object`
- `Type B2 / MDI / multi_hop_direction`
- `Type C1 / PTM / move_translation`
- `Type C2 / RTM / move_turn_combined`

## Highlights

- End-to-end QA generation from structured scene JSON
- Externalized prompt system under `prompts/`
- Scene-level multiprocessing
- Cache-based resume support
- Deterministic random object sampling via configurable seed

## Repository Structure

```text
Omni-COT/
|- run.py
|- config/
|  |- api_config.yaml
|  `- batch_config.yaml
|- prompts/
|- src/
|- data/
|  |- cache/
|  `- outputs/
|- requirements.txt
`- .gitignore
```

## Installation

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
# source .venv/bin/activate

pip install -r requirements.txt
```

## Quick Start

1. Configure your API endpoint and model names in `config/api_config.yaml`.
2. Prepare a dataset root where each scene is stored in its own folder and includes `scene_data.json`.
3. Run the pipeline:

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

Check all CLI options with:

```bash
python run.py --help
```

## OmniCoT Design and Generation

### Question Dimension: "See-Locate-Move"

Inspired by real-world embodied requirements, OmniCoT designs complex panoramic spatial reasoning questions from three progressive dimensions: `See`, `Locate`, and `Move`.

- `See` focuses on viewpoint transformation. It tests whether a model can decouple its view from the camera's initial pose, mentally synthesize views from arbitrary angles, and reason through spherical panoramic geometry.
- `Locate` advances to spatial object relationships. It evaluates whether a model can reason over inter-object spatial structure rather than treating the scene as an unstructured collection of pixels.
- `Move` introduces embodied action simulation. It tests whether a model can execute virtual movements and predict the visual consequences of those actions for embodied downstream scenarios.

### [See] Multi-hop Viewpoint Transformation

This category evaluates the model's ability to observe the world through multi-step panorama reasoning and maintain spatial constancy under viewpoint change.

- `Type A1: Multi-Step Orientation Tracking (MOT)` corresponds to `viewpoint_transform_identify`. The model starts from an initial pose, executes a sequence of relative rotations, and identifies the target object.
- `Type A2: Relative Angular Calculation (RAC)` corresponds to `viewpoint_transform_angle`. The model computes a cumulative angular displacement or bearing across multiple landmarks.

Example templates:

- `Type A1`: Standing at the `[object_desc]`, facing `[cardinal direction]`, turn `[angle1] [dir1]`, then turn `[angle2] [dir2]`, what is the nearest object?
- `Type A2`: Standing at `[object_desc]`, initially facing `[object_desc]`, turn to face `[object_desc]`, then turn to face `[object_desc]`. What is the total cumulative angle turned?

### [Locate] Inter-Object Spatial Relationship

This category evaluates the model's ability to reason about spatial relationships between multiple objects through multi-hop relational chains.

- `Type B1: Multi-Hop Object Identification (MOI)` corresponds to `multi_hop_object`. The model sequentially applies spatial qualifiers to identify a unique target object.
- `Type B2: Multi-Hop Direction Identification (MDI)` corresponds to `multi_hop_direction`. The model determines the final direction of a target relative to an anchor after traversing intermediate references.

Example templates:

- `Type B1`: What `[object_desc]` is directly to the `[cardinal direction]` of the nearest object that is `[cardinal direction]` of the `[object_desc]`?
- `Type B2`: In which direction is the `[object_desc]` to the `[cardinal direction]` of the `[object_desc]`, relative to the `[object_desc]` itself?

### [Move] Embodied Action Simulation

This category evaluates dynamic spatial reasoning under virtual movement and turning, including updates to position, orientation, and field of view.

- `Type C1: Pure Translational Movement (PTM)` corresponds to `move_translation`. The model predicts objects encountered along a linear path without rotation.
- `Type C2: Rotation-Translational Movement (RTM)` corresponds to `move_turn_combined`. The model updates its egocentric perspective after movement and rotation, then reasons about visibility or altered spatial relations.

Example templates:

- `Type C1`: From the `[object_desc]` near the `[object_desc]`, walk straight `[cardinal direction]` for `[number]` meters. What is the first object you will encounter?
- `Type C2`: From the `[object_desc]`, walk `[cardinal direction]` `[number]` meters toward the `[object_desc]` area, then turn `[angle]` to face `[cardinal direction]`. Is the `[object_desc]` still visible from your new position and facing direction?

### Data Generation

#### Step 1: Question-Answer Generation

Following the paper design, the generation pipeline starts from reliable raw 3D scene data and transforms it into a structured language representation. This representation provides spatial context for subsequent question generation and enables questions that leverage the global perspective of panoramic scenes.

On top of this representation, the pipeline generates multiple batches of candidate questions covering all predefined question types. Candidate questions are then scored by dual LLM judges across dimensions such as:

- format compliance
- object uniqueness
- logical consistency
- reasoning complexity
- answerability
- type-specific auto-penalties

In the paper setup, the dual-judge stage is instantiated with DeepSeekv3.2 and Qwen3-Max. In the open-source release, the corresponding judge models are configured through `provider.models.reasoning` and `provider.models.text`, so the same pipeline can be reproduced with compatible model choices.

#### Step 2: Chain-of-Thought Generation

Building on validated QA pairs, OmniCoT uses a structured CoT generation pipeline to produce step-by-step spatial reasoning traces.

- First, type-specific prompting generates reasoning traces with 2-4 clear reasoning steps, appropriate transition words, and strict natural-language compliance.
- Then, a summarization stage distills the core logical flow while preserving critical spatial information.
- Finally, a quality-evaluation stage scores both the reasoning process and the final answer across multiple dimensions, including format compliance, reasoning structure, scene information utilization, type-specific penalties, and answer correctness.

In the paper workflow, accepted QA-CoT pairs undergo expert review, and a random subset is carefully evaluated until the quality target is satisfied. The released codebase covers the automatic generation, scoring, caching, and export stages; manual expert verification remains a dataset curation step outside the code.

## Configuration

### API config

Edit `config/api_config.yaml` directly, or create a separate local config file and pass it with `--config`.

Required fields:

- `provider.api_key` or `${OPENAI_API_KEY}`
- `provider.base_url` or `${OPENAI_BASE_URL}`
- `provider.models.vision`
- `provider.models.reasoning`
- `provider.models.text`

Key runtime options:

- `generation.temperature.*`: per-stage temperatures
- `generation.max_tokens.*`: per-stage token limits
- `generation.max_retries`: retry count for model calls
- `generation.timeout`: per-request timeout in seconds
- `generation.stream`: whether generation uses streaming responses

### Batch config

`config/batch_config.yaml` currently controls:

- `batch.random_seed`: base seed for reproducible random object sampling
- `batch.cache.stage_cache_dir`: stage cache directory

The default configuration uses `batch.random_seed: 42`.

## Data Format

`--data-root` should contain one subdirectory per scene:

```text
path/to/osr_scenes/
|- scene_0001/
|  `- scene_data.json
|- scene_0002/
|  `- scene_data.json
`- ...
```

### Minimum `scene_data.json` example

The current pipeline requires the fields used by `src/scene_data_extractor.py`: camera position, room layout, and object list with 3D boxes.

```json
{
  "name": "scene_0001",
  "camera": {
    "pos": [0.0, 0.0, 1.6]
  },
  "layout": {
    "manhattan_world": [
      [-2.0, -2.0, 0.0],
      [2.0, -2.0, 0.0],
      [2.0, 2.0, 0.0],
      [-2.0, 2.0, 0.0],
      [-2.0, -2.0, 2.4],
      [2.0, -2.0, 2.4],
      [2.0, 2.0, 2.4],
      [-2.0, 2.0, 2.4]
    ]
  },
  "objs": [
    {
      "id": 1,
      "classname": "chair",
      "bdb3d": {
        "centroid": [0.8, -0.5, 0.5],
        "size": [0.6, 0.6, 1.0],
        "basis": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      }
    },
    {
      "id": 2,
      "classname": "table",
      "bdb3d": {
        "centroid": [-0.7, 0.9, 0.75],
        "size": [1.2, 0.8, 0.75],
        "basis": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
      }
    }
  ]
}
```

Notes:

- `layout.manhattan_world` is expected to contain floor and ceiling vertices; the extractor currently uses the first half as floor boundary vertices.
- Objects without `bdb3d` are ignored.
- The open-source pipeline currently operates on structured geometry only and does not require RGB images.

## Pipeline Stages

For each scene, the released code performs:

1. Structured scene extraction from `scene_data.json`
2. Scene-text preparation
3. Multi-type question generation
4. Dual-judge question scoring and filtering
5. CoT and answer generation
6. Dual-judge QA quality evaluation

Implementation note:

- `SceneUnderstanding` in the current open-source version is a deterministic passthrough stage. It forwards the structured scene text produced by `SceneDataExtractor` instead of calling a separate scene-understanding model.

## Prompt System

All prompts are loaded from `prompts/` via `src/prompt_loader.py`.

Main prompt groups:

- `prompts/question_generator/`
- `prompts/question_scorer/`
- `prompts/cot_generator/`
- `prompts/quality_judge/`

The prompt taxonomy uses both the released fine-grained types and a smaller set of mapped reasoning categories for scoring and answer generation:

- `viewpoint_transform_identify` and `viewpoint_transform_angle` map to `viewpoint_transform`
- `multi_hop_object` and `multi_hop_direction` map to `multi_step_viewpoint`
- `move_translation` and `move_turn_combined` map to `move_translation`

## Output Files

### Raw pipeline output

The file passed to `--output` stores the accepted QA items returned directly by `run.py`. Each item has the following structure:

```json
{
  "scene_id": "scene_0001",
  "question_data": {
    "question": "Standing at the chair near the table, facing north, turn 90 degrees right, then turn 45 degrees left. What is the nearest object directly ahead?",
    "type": "viewpoint_transform_identify"
  },
  "reasoning_process": {
    "initial_reasoning": "...",
    "structured_cot": "..."
  },
  "answer_data": {
    "final_answer": "the table",
    "cot": [
      "First, ...",
      "Next, ...",
      "Finally, ..."
    ]
  },
  "quality_evaluation": {
    "reasoning_score": 7.5,
    "answer_score": 8.0,
    "overall_score": 7.75,
    "passed": true
  },
  "token_usage": {
    "total": 1234
  },
  "timestamp": "2026-03-29T10:00:00"
}
```

### Exported dataset format

If you later export from stage cache via `src/stage_cache.py`, the exported schema is different and more dataset-oriented:

```bash
python src/stage_cache.py --cache-dir data/cache/stage_cache --export data/outputs/exported_qa.json
```

That export includes fields such as:

- `image`
- `QA_id`
- `type`
- `scene_id`
- `description`
- `subtype`
- `question`
- `answer`
- `CoT`
- `Steps`
- `random_objects`

If you plan to publish exported QA files, review these fields carefully before release.

## Cache and Resume

Default cache locations:

- `data/cache/question_scores`
- `data/cache/quality_evaluations`
- `data/cache/stage_cache`

If the run is interrupted, rerun the same command to resume from cache.

Useful cache commands:

```bash
python src/stage_cache.py --cache-dir data/cache/stage_cache --stats
python src/stage_cache.py --cache-dir data/cache/stage_cache --list
python src/stage_cache.py --cache-dir data/cache/stage_cache --scene scene_0001
```



