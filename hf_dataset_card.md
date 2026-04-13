---
license: mit
task_categories:
  - object-detection
  - visual-question-answering
language:
  - en
tags:
  - multi-object-tracking
  - referring-expression
  - CARLA
  - autonomous-driving
  - text-guided-tracking
  - synthetic
pretty_name: CARLA Referring Target Evaluation Set
size_categories:
  - 10K<n<100K
---

# CARLA Referring Target Evaluation Set

A synthetic benchmark for **referring expression tracking** — the task of tracking a single described target (e.g. *"red sedan"*) while ignoring all other vehicles in the scene.

Unlike standard MOT datasets that evaluate all objects equally, this benchmark tests whether a tracker can isolate and continuously follow exactly the object that matches a natural language description, across challenging conditions including weather variation, lighting changes, camera motion, and the presence of visually similar distractors.

Generated using the [CARLA simulator](https://carla.org/) (Towns 10HD, 03, 05).

## Dataset Structure

The dataset is provided as a single zip file: `eval_scenarios.zip`.

After extracting, the structure is:

```
eval_scenarios/
  master_index.json          # metadata for all 24 scenarios
  clear_day_baseline/
    images/                  # per-frame PNG images (1920×1080)
    labels/                  # per-frame YOLO-style labels with is_target flags
    gt.json                  # ground truth with prompt, bboxes, target flags
  overcast/
  heavy_rain/
  ...
```

Each `gt.json` contains:
- `prompt`: the referring expression (e.g. `"red sedan"`)
- per-frame bounding boxes for all vehicles
- `is_target: true/false` flag distinguishing the referred object from distractors

## Scenarios

| # | Name | Camera | Condition |
|---|------|--------|-----------|
| 1 | `clear_day_baseline` | static | Clear sky, high sun |
| 2 | `overcast` | static | Heavy cloud cover |
| 3 | `heavy_rain` | loose-follow | Rain, wet road reflections |
| 4 | `dusk_golden_hour` | loose-follow | Low sun, warm directional light |
| 5 | `night` | loose-follow | Night lighting |
| 6 | `dense_fog` | loose-follow | Low visibility fog |
| 7 | `color_confusable` | loose-follow | Other red vehicles as distractors |
| 8 | `same_color_diff_class` | loose-follow | Same color, different vehicle class |
| 9 | `high_density` | static | Dense mixed traffic |
| 10 | `high_altitude` | static | Elevated camera angle |
| 11 | `low_altitude_steep` | loose-follow | Low steep camera angle |
| 12 | `side_follow` | loose-follow | Side-on perspective |
| 13 | `town03_suburban` | static | Suburban map (Town03) |
| 14 | `town05_highway` | loose-follow | Highway map (Town05) |
| 15 | `multiple_red_sedans` | static | 3 target vehicles simultaneously |
| 16 | `long_sequence` | static | Extended duration |
| 17 | `dense_urban_traffic` | static | Dense urban scene |
| 18–24 | `follow_base` / `follow_variant_*` | follow | Follow-camera variants |

## Evaluation Metrics

Evaluated with the companion `eval_carla.py` script:

| Metric | Description |
|--------|-------------|
| **Semantic Precision** | Fraction of tracked detections that are actually the target |
| **Semantic Recall** | Fraction of target frames where the target was tracked |
| **Prompt Coverage Ratio** | How consistently the described object is tracked overall |
| **Distractor Confusion Rate** | How often a non-target was incorrectly tracked as the target |
| **Semantic ID Switches (SID)** | Times the tracker switched from the correct target to a distractor |

## Usage

```bash
# Download
hf download azzy13/carla_referring_target_evaluation_set eval_scenarios.zip \
  --repo-type dataset --local-dir dataset/carla_eval/

unzip dataset/carla_eval/eval_scenarios.zip -d dataset/carla_eval/

# Evaluate
python3 eval/eval_carla.py \
  --carla_scenarios dataset/carla_eval/eval_scenarios \
  --tracker clip --fp16
```

## Citation

```bibtex
@article{hasan2024textguidedmot,
  title={Text-Guided Multi-Object Tracking},
  author={Hasan, Azhar and Karsai, Gabor},
  year={2024},
  institution={Vanderbilt University}
}
```
