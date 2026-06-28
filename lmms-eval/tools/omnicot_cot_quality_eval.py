#!/usr/bin/env python3
"""Evaluate OmniCoT reasoning quality from lmms-eval submissions.

This is a post-processing judge pipeline. Run lmms-eval first to produce
omnicot_submission.json, then run this script on that file.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from tqdm import tqdm

    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False


DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_JUDGE_MODEL = "gpt-4o-mini"

REQUEST_TIMEOUT = float(os.environ.get("REQUEST_TIMEOUT", "120"))
MAX_RETRIES = int(os.environ.get("MAX_RETRIES", "3"))
BACKOFF_BASE = float(os.environ.get("BACKOFF_BASE", "1.5"))
BACKOFF_MAX = float(os.environ.get("BACKOFF_MAX", "30"))
JITTER = float(os.environ.get("JITTER", "0.3"))
REQUEST_SLEEP = float(os.environ.get("REQUEST_SLEEP", "0"))

MAX_GT_CHARS = int(os.environ.get("MAX_GT_CHARS", "12000"))
MAX_REASONING_CHARS = int(os.environ.get("MAX_REASONING_CHARS", "6000"))
MAX_REF_COT_CHARS = int(os.environ.get("MAX_REF_COT_CHARS", "6000"))


PRECISION_PROMPT = """# Task Overview
Given a solution with multiple reasoning steps for an image-based problem,
reformat it into well-structured steps and evaluate their correctness.

# Step Types
1. Logical Inference Steps
   - Contains exactly one logical deduction.
   - Must produce a new derived conclusion.
2. Image Description Steps
   - Pure visual observations.
   - Only includes directly visible elements.
3. Background Information Steps
   - External knowledge or question context.
   - No inference process involved.

# Judgement Categories
- "Match": Aligns with the reference reasoning or ground truth.
- "Reasonable": Valid but not explicitly in the reference reasoning.
- "Wrong": Invalid or contradictory.
- "N/A": Background information steps.

# Output Requirements
Output ONLY valid JSON. Maximum 35 steps. Always include the final step that
contains the answer.

JSON schema:
[
  {{
    "step_type": "image description|logical inference|background information",
    "premise": "Evidence, only for logical inference",
    "conclusion": "Step result",
    "judgment": "Match|Reasonable|Wrong|N/A"
  }}
]

[Problem]
{question}

[Solution]
{reasoning}

[Correct Answer]
{answer}

[Reference Reasoning]
{reference_cot}

[Ground Truth]
{ground_truth}
"""


RECALL_PROMPT = """You are an expert system for verifying solutions to
image-based problems. Match each reference reasoning step with the provided
solution.

Match Criteria:
- The reference step should exactly match or be directly entailed by content in
  the solution.
- All specific values and entities must match.
- Judge every reference step without omitting any step.

Output ONLY valid JSON:
[
  {{
    "step_index": 0,
    "judgment": "Matched|Unmatched"
  }}
]

[Problem]
{question}

[Answer]
{answer}

[Solution]
{reasoning}

[Reference Reasoning]
{reference_cot}
"""


VIEWPOINT_PROMPT = """You are an expert in evaluating spatial reasoning quality.

Task: Evaluate viewpoint consistency in the reasoning.

Definitions:
- "Viewpoint-related statement" is a claim about directions, orientation, or
  relative position that depends on the agent pose: left/right, front/back,
  facing, turn, behind, clockwise, north/east, etc.

Rules:
1. Extract viewpoint-related statements from the reasoning. Prefer statements
   used to support the final answer.
2. Do not list mere restatements of the question instruction unless the
   reasoning relies on them to infer geometry.
3. Merge duplicates.
4. If no viewpoint-related statements are found, return {{"statements": []}}.

Scoring:
- score is a float in [0.0, 1.0] with one decimal place.
- 0.0: clear contradiction.
- 0.1-0.4: mostly incorrect or highly ambiguous.
- 0.5-0.8: mostly consistent, minor underspecification.
- 0.9-1.0: fully consistent and precise.

[Ground Truth]
{ground_truth}

[Question]
{question}

[Reasoning to Evaluate]
{reasoning}

[Reference Reasoning]
{reference_cot}

[Correct Answer]
{answer}

Output ONLY valid JSON:
{{
  "statements": [
    {{ "text": "...", "score": 0.9 }}
  ]
}}
"""


SPATIAL_EVIDENCE_PROMPT = """You are an expert in evaluating spatial reasoning
quality.

Task: Evaluate spatial evidence sufficiency.

Goal:
Identify the minimal set of key spatial relationships necessary to justify the
conclusion for this question, then check whether the reasoning provides evidence
for each relationship.

Rules:
1. List only the minimal necessary relationships.
2. Recognize alternative valid reasoning paths; reference reasoning is only a
   hint, not a required path.
3. Evidence must be a direct quote or near-quote from the reasoning. If no quote
   exists, set score=0.0 and evidence=null.
4. If the reasoning contains no usable spatial evidence, return
   {{"relationships": []}}.

Scoring:
- score is a float in [0.0, 1.0] with one decimal place.
- 0.0: unsupported by reasoning or no evidence quote.
- 0.1-0.4: weak or partial.
- 0.5-0.8: clearly supported but slightly underspecified.
- 0.9-1.0: clearly and sufficiently supported.

[Ground Truth]
{ground_truth}

[Question]
{question}

[Reference Reasoning]
{reference_cot}

[Reasoning to Evaluate]
{reasoning}

Output ONLY valid JSON:
{{
  "relationships": [
    {{ "text": "...", "score": 0.7, "evidence": "..." }}
  ]
}}
"""


FEASIBILITY_PROMPT = """You are an expert in evaluating spatial reasoning
quality.

Task: Evaluate reasoning feasibility by extracting execution steps and judging
whether they are feasible in the scene.

Rules:
1. Extract execution or inference steps. Cap to 10 steps.
2. Penalize hard spatial impossibilities:
   - unknown objects or locations: score <= 0.2
   - through walls or barriers: score <= 0.2
   - impossible move or turn: score <= 0.2
3. If no execution steps are found, return {{"steps": []}}.

Scoring:
- score is a float in [0.0, 1.0] with one decimal place.

[Ground Truth]
{ground_truth}

[Question]
{question}

[Reasoning to Evaluate]
{reasoning}

[Correct Answer]
{answer}

[Reference Reasoning]
{reference_cot}

Output ONLY valid JSON:
{{
  "steps": [
    {{ "text": "...", "score": 0.9 }}
  ]
}}
"""


_thread_local = threading.local()
_checkpoint_lock = threading.Lock()


def truncate_text(value: Any, max_chars: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + f"\n...[TRUNCATED {len(text) - max_chars} chars]..."


def first_value(item: Dict[str, Any], *keys: str, default: Any = "") -> Any:
    for key in keys:
        value = item.get(key)
        if value not in (None, ""):
            return value
    return default


def as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def extract_reasoning(response: Any) -> str:
    text = str(response or "").strip()
    match = re.search(r"<think>\s*(.*?)\s*</think>", text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()
    parts = re.split(r"<answer>", text, flags=re.IGNORECASE)
    if len(parts) > 1:
        return re.sub(r"</think>\s*$", "", parts[0].strip(), flags=re.IGNORECASE)
    return text


def load_ground_truth_from_dir(scene_id: str, gt_dir: Optional[str]) -> str:
    if not scene_id or not gt_dir:
        return ""
    root = Path(gt_dir)
    if not root.exists():
        return ""

    exact = root / f"{scene_id}.txt"
    candidates = [exact] if exact.exists() else list(root.glob(f"{scene_id}*"))
    for candidate in candidates:
        if candidate.is_file():
            try:
                return candidate.read_text(encoding="utf-8")
            except Exception:
                return ""
    return ""


def normalize_item(item: Dict[str, Any], index: int, gt_dir: Optional[str]) -> Dict[str, Any]:
    qa_id = first_value(item, "QA_id", "qa_id", "item_id", "id", default=f"__index_{index}")
    prediction = first_value(item, "prediction", "response", "model_response", default="")
    reasoning = first_value(item, "reasoning", default="")
    if not reasoning and prediction:
        reasoning = extract_reasoning(prediction)

    reference_cot = first_value(item, "reference_cot", "cot", "CoT", default=[])
    scene_id = first_value(item, "scene_id", "image_id", default="")
    ground_truth = first_value(item, "ground_truth", "description", default="")
    ground_truth_source = "submission"
    if not ground_truth:
        ground_truth = load_ground_truth_from_dir(str(scene_id), gt_dir)
        ground_truth_source = "gt_dir" if ground_truth else ""
    if not ground_truth and reference_cot:
        ground_truth = reference_cot
        ground_truth_source = "reference_cot"

    return {
        "QA_id": str(qa_id),
        "qa_id": str(qa_id),
        "scene_id": str(scene_id),
        "question": first_value(item, "question", default=""),
        "answer": first_value(item, "answer", "target", default=""),
        "prediction": prediction,
        "pred_extracted": first_value(item, "pred_extracted", "pred", default=""),
        "reasoning": reasoning,
        "reference_cot": reference_cot,
        "ground_truth": ground_truth,
        "ground_truth_source": ground_truth_source,
        "type": first_value(item, "type", default=""),
        "subtype": first_value(item, "subtype", default=""),
        "is_correct": item.get("is_correct"),
        "is_contains_correct": item.get("is_contains_correct"),
    }


def strip_code_fence(text: str) -> str:
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def find_first_json_span(text: str) -> Optional[Tuple[int, int]]:
    starts = [(text.find("{"), "{"), (text.find("["), "[")]
    starts = [(idx, char) for idx, char in starts if idx != -1]
    if not starts:
        return None

    start, opening = min(starts, key=lambda pair: pair[0])
    closing = "}" if opening == "{" else "]"
    depth = 0
    in_string = False
    escaped = False

    for pos in range(start, len(text)):
        char = text[pos]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return start, pos + 1

    return None


def extract_json(text: str) -> Any:
    candidates = [strip_code_fence(text), text]
    span = find_first_json_span(candidates[0])
    if span:
        candidates.append(candidates[0][span[0] : span[1]])
    span = find_first_json_span(text)
    if span:
        candidates.append(text[span[0] : span[1]])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, str):
                parsed = json.loads(parsed)
            if isinstance(parsed, (dict, list)):
                return parsed
        except Exception:
            continue

    raise ValueError(f"No valid JSON found. Sample: {text[:300]}")


def should_retry(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        signal in text
        for signal in (
            "rate limit",
            "429",
            "timeout",
            "timed out",
            "connection reset",
            "connection aborted",
            "server error",
            "502",
            "503",
            "504",
            "temporarily unavailable",
        )
    )


def get_client():
    client = getattr(_thread_local, "client", None)
    if client is not None:
        return client

    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    base_url = os.environ.get("OPENAI_BASE_URL", DEFAULT_BASE_URL)
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=REQUEST_TIMEOUT)
    _thread_local.client = client
    return client


def query_judge(messages: List[Dict[str, str]], max_tokens: int = 4096) -> Optional[str]:
    model = os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL)
    client = get_client()
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if REQUEST_SLEEP > 0:
                time.sleep(REQUEST_SLEEP)
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
                top_p=0.1,
            )
            return response.choices[0].message.content
        except Exception as exc:
            last_error = exc
            if attempt >= MAX_RETRIES or not should_retry(exc):
                break
            delay = min(BACKOFF_MAX, BACKOFF_BASE ** (attempt - 1))
            delay *= 1.0 + random.uniform(-JITTER, JITTER)
            time.sleep(max(0.1, delay))

    print(f"[WARNING] Judge API failed after {MAX_RETRIES} tries: {last_error}")
    return None


def make_messages(prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "You are an expert in reasoning quality evaluation."},
        {"role": "user", "content": prompt},
    ]


def prompt_inputs(item: Dict[str, Any]) -> Dict[str, str]:
    return {
        "question": truncate_text(item.get("question", ""), 3000),
        "answer": truncate_text(item.get("answer", ""), 500),
        "reasoning": truncate_text(item.get("reasoning", ""), MAX_REASONING_CHARS),
        "reference_cot": truncate_text(item.get("reference_cot", []), MAX_REF_COT_CHARS),
        "ground_truth": truncate_text(item.get("ground_truth", ""), MAX_GT_CHARS),
    }


def evaluate_precision(item: Dict[str, Any]) -> Dict[str, Any]:
    response = query_judge(make_messages(PRECISION_PROMPT.format(**prompt_inputs(item))), max_tokens=8192)
    if not response:
        return {"precision": [], "precision_error": "API call failed"}
    try:
        parsed = extract_json(response)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected list, got {type(parsed).__name__}")
        return {"precision": parsed}
    except Exception as exc:
        return {"precision": [], "precision_error": f"JSON parsing failed: {exc}"}


def evaluate_recall(item: Dict[str, Any]) -> Dict[str, Any]:
    response = query_judge(make_messages(RECALL_PROMPT.format(**prompt_inputs(item))), max_tokens=4096)
    if not response:
        return {"recall": [], "recall_error": "API call failed"}
    try:
        parsed = extract_json(response)
        if not isinstance(parsed, list):
            raise ValueError(f"Expected list, got {type(parsed).__name__}")
        matched = sum(1 for step in parsed if isinstance(step, dict) and step.get("judgment") == "Matched")
        total = len(parsed)
        return {
            "recall": parsed,
            "recall_score": matched / total if total else None,
            "matched_count": matched,
            "total_count": total,
        }
    except Exception as exc:
        return {"recall": [], "recall_error": f"JSON parsing failed: {exc}"}


def average_scores(items: Iterable[Dict[str, Any]], key: str) -> Optional[float]:
    values = []
    for item in items:
        try:
            values.append(float(item.get(key, 0.0)))
        except Exception:
            continue
    return sum(values) / len(values) if values else None


def evaluate_viewpoint_consistency(item: Dict[str, Any]) -> Dict[str, Any]:
    response = query_judge(make_messages(VIEWPOINT_PROMPT.format(**prompt_inputs(item))), max_tokens=4096)
    if not response:
        return {"viewpoint_consistency": {"consistency_score": None, "error": "API call failed"}}
    try:
        parsed = extract_json(response)
        statements = parsed.get("statements", []) if isinstance(parsed, dict) else []
        if not isinstance(statements, list):
            raise ValueError("'statements' is not a list")
        return {
            "viewpoint_consistency": {
                "raw_output": statements,
                "consistency_score": average_scores(statements, "score"),
                "num_statements": len(statements),
            }
        }
    except Exception as exc:
        return {"viewpoint_consistency": {"consistency_score": None, "error": f"JSON parsing failed: {exc}"}}


def evaluate_spatial_evidence(item: Dict[str, Any]) -> Dict[str, Any]:
    response = query_judge(make_messages(SPATIAL_EVIDENCE_PROMPT.format(**prompt_inputs(item))), max_tokens=4096)
    if not response:
        return {"spatial_evidence": {"sufficiency_score": None, "error": "API call failed"}}
    try:
        parsed = extract_json(response)
        relationships = parsed.get("relationships", []) if isinstance(parsed, dict) else []
        if not isinstance(relationships, list):
            raise ValueError("'relationships' is not a list")

        num_with_evidence = 0
        for rel in relationships:
            if not isinstance(rel, dict):
                continue
            evidence = rel.get("evidence")
            has_evidence = evidence not in (None, "", "null", "none")
            try:
                score = float(rel.get("score", 0.0))
            except Exception:
                score = 0.0
            if score > 0.0 and not has_evidence:
                rel["score"] = 0.0
                rel["evidence"] = None
            elif has_evidence:
                num_with_evidence += 1

        total = len(relationships)
        return {
            "spatial_evidence": {
                "raw_output": relationships,
                "sufficiency_score": average_scores(relationships, "score"),
                "num_relationships": total,
                "num_with_evidence": num_with_evidence,
                "evidence_rate": num_with_evidence / total if total else None,
            }
        }
    except Exception as exc:
        return {"spatial_evidence": {"sufficiency_score": None, "error": f"JSON parsing failed: {exc}"}}


def evaluate_reasoning_feasibility(item: Dict[str, Any]) -> Dict[str, Any]:
    response = query_judge(make_messages(FEASIBILITY_PROMPT.format(**prompt_inputs(item))), max_tokens=4096)
    if not response:
        return {"reasoning_feasibility": {"feasibility_score": None, "error": "API call failed"}}
    try:
        parsed = extract_json(response)
        steps = parsed.get("steps", []) if isinstance(parsed, dict) else []
        if not isinstance(steps, list):
            raise ValueError("'steps' is not a list")
        feasible = 0
        for step in steps:
            if not isinstance(step, dict):
                continue
            try:
                feasible += 1 if float(step.get("score", 0.0)) >= 0.5 else 0
            except Exception:
                pass
        total = len(steps)
        return {
            "reasoning_feasibility": {
                "raw_output": steps,
                "feasibility_score": average_scores(steps, "score"),
                "num_steps": total,
                "num_feasible": feasible,
                "feasibility_rate": feasible / total if total else None,
            }
        }
    except Exception as exc:
        return {"reasoning_feasibility": {"feasibility_score": None, "error": f"JSON parsing failed: {exc}"}}


def simple_item_scores(result: Dict[str, Any]) -> Dict[str, Any]:
    target_types = {"logical inference", "image description"}
    precision_steps = result.get("precision", [])
    precision_total = 0
    precision_ok = 0
    precision_match = 0
    for step in precision_steps if isinstance(precision_steps, list) else []:
        if not isinstance(step, dict):
            continue
        step_type = str(step.get("step_type", "")).lower()
        judgment = step.get("judgment")
        if step_type not in target_types:
            continue
        precision_total += 1
        if judgment in ("Match", "Reasonable"):
            precision_ok += 1
        if judgment == "Match":
            precision_match += 1

    recall_steps = result.get("recall", [])
    recall_total = len(recall_steps) if isinstance(recall_steps, list) else 0
    recall_matched = sum(
        1
        for step in recall_steps
        if isinstance(step, dict) and step.get("judgment") == "Matched"
    )

    precision_score = precision_ok / precision_total if precision_total else None
    match_ratio = precision_match / precision_ok if precision_ok else None
    recall_score = recall_matched / recall_total if recall_total else None
    if precision_score is not None and recall_score is not None and precision_score + recall_score > 0:
        f1 = 2 * precision_score * recall_score / (precision_score + recall_score)
    else:
        f1 = None

    return {
        "precision_score": precision_score,
        "precision_match_ratio": match_ratio,
        "precision_target_steps": precision_total,
        "recall_score": recall_score,
        "recall_matched": recall_matched,
        "recall_total": recall_total,
        "f1_score": f1,
    }


def evaluate_single_item(item: Dict[str, Any], mode: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "item_id": item["QA_id"],
        "QA_id": item["QA_id"],
        "qa_id": item["qa_id"],
        "scene_id": item.get("scene_id", ""),
        "type": item.get("type", ""),
        "subtype": item.get("subtype", ""),
        "is_correct": item.get("is_correct"),
        "is_contains_correct": item.get("is_contains_correct"),
        "ground_truth_source": item.get("ground_truth_source", ""),
    }

    if mode in ("simple", "all"):
        result.update(evaluate_precision(item))
        result.update(evaluate_recall(item))
        result["simple_scores"] = simple_item_scores(result)

    if mode in ("spatial", "all"):
        if not item.get("ground_truth"):
            missing = {"error": "Missing ground truth and reference CoT"}
            result["viewpoint_consistency"] = {"consistency_score": None, **missing}
            result["spatial_evidence"] = {"sufficiency_score": None, **missing}
            result["reasoning_feasibility"] = {"feasibility_score": None, **missing}
        else:
            result.update(evaluate_viewpoint_consistency(item))
            result.update(evaluate_spatial_evidence(item))
            result.update(evaluate_reasoning_feasibility(item))

    return result


def item_key(item: Dict[str, Any], index: int) -> str:
    return str(first_value(item, "QA_id", "qa_id", "item_id", "id", default=f"__index_{index}"))


def result_key(result: Dict[str, Any]) -> str:
    return str(first_value(result, "__item_key", "item_id", "QA_id", "qa_id", default=""))


def strip_internal_keys(result: Dict[str, Any]) -> Dict[str, Any]:
    return {key: value for key, value in result.items() if not key.startswith("__")}


def load_input(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        records = []
        with path.open("r", encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("submission", "samples", "results"):
            value = data.get(key)
            if isinstance(value, list):
                return value
        if all(isinstance(value, list) for value in data.values()):
            merged = []
            for value in data.values():
                merged.extend(value)
            return merged
    raise ValueError(f"Unsupported input structure in {path}")


def load_cached_results(checkpoint_file: Path, final_file: Path) -> Dict[str, Dict[str, Any]]:
    cached: Dict[str, Dict[str, Any]] = {}
    if final_file.exists():
        try:
            for result in json.loads(final_file.read_text(encoding="utf-8")):
                key = result_key(result)
                if key:
                    cached[key] = result
        except Exception as exc:
            print(f"[WARNING] Failed to read existing final results: {exc}")

    if checkpoint_file.exists():
        with checkpoint_file.open("r", encoding="utf-8") as file:
            for line_no, line in enumerate(file, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    result = json.loads(line)
                except json.JSONDecodeError as exc:
                    print(f"[WARNING] Skip broken checkpoint line {line_no}: {exc}")
                    continue
                key = result_key(result)
                if key:
                    cached[key] = result
    return cached


def append_checkpoint(checkpoint_file: Path, result: Dict[str, Any]) -> None:
    with _checkpoint_lock:
        with checkpoint_file.open("a", encoding="utf-8") as file:
            file.write(json.dumps(result, ensure_ascii=False) + "\n")
            file.flush()
            os.fsync(file.fileno())


def mean(values: Iterable[Optional[float]]) -> Optional[float]:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def format_score(value: Optional[float]) -> Optional[str]:
    return f"{value:.4f}" if value is not None else None


def score_from_path(result: Dict[str, Any], path: List[str]) -> Optional[float]:
    current: Any = result
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    try:
        return float(current) if current is not None else None
    except Exception:
        return None


def generate_report(results: List[Dict[str, Any]], mode: str, output_dir: Path) -> Dict[str, Any]:
    answer_values = [result.get("is_correct") for result in results if result.get("is_correct") is not None]
    contains_values = [
        result.get("is_contains_correct")
        for result in results
        if result.get("is_contains_correct") is not None
    ]

    simple_precision = mean(
        score_from_path(result, ["simple_scores", "precision_score"]) for result in results
    )
    simple_recall = mean(score_from_path(result, ["simple_scores", "recall_score"]) for result in results)
    simple_f1 = mean(score_from_path(result, ["simple_scores", "f1_score"]) for result in results)

    vc = mean(score_from_path(result, ["viewpoint_consistency", "consistency_score"]) for result in results)
    se = mean(score_from_path(result, ["spatial_evidence", "sufficiency_score"]) for result in results)
    rf = mean(score_from_path(result, ["reasoning_feasibility", "feasibility_score"]) for result in results)

    spatial_scores = [value for value in (vc, se, rf) if value is not None]
    spatial_overall = sum(spatial_scores) / len(spatial_scores) if spatial_scores else None

    report = {
        "overview": {
            "total_samples": len(results),
            "mode": mode,
            "judge_model": os.environ.get("JUDGE_MODEL", DEFAULT_JUDGE_MODEL),
        },
        "answer_accuracy": {
            "accuracy": format_score(sum(1 for value in answer_values if value) / len(answer_values))
            if answer_values
            else None,
            "contains_accuracy": format_score(
                sum(1 for value in contains_values if value) / len(contains_values)
            )
            if contains_values
            else None,
        },
        "cot_precision_recall": {
            "precision": format_score(simple_precision),
            "recall": format_score(simple_recall),
            "f1": format_score(simple_f1),
        },
        "spatial_reasoning_quality": {
            "viewpoint_consistency": format_score(vc),
            "spatial_evidence_sufficiency": format_score(se),
            "reasoning_feasibility": format_score(rf),
            "overall": format_score(spatial_overall),
        },
        "notes": {
            "cot_precision": "Mean per-sample (Match + Reasonable) / target reasoning steps.",
            "cot_recall": "Mean per-sample coverage of reference CoT steps.",
            "spatial_quality": "LLM-judge scores in [0, 1]; ground_truth_source records whether descriptions or reference CoT were used.",
        },
    }

    report_file = output_dir / "cot_quality_report.json"
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nCoT Quality Report")
    print("==================")
    print(f"Samples: {len(results)}")
    if mode in ("simple", "all"):
        print(f"Precision: {format_score(simple_precision)}")
        print(f"Recall:    {format_score(simple_recall)}")
        print(f"F1:        {format_score(simple_f1)}")
    if mode in ("spatial", "all"):
        print(f"VC:        {format_score(vc)}")
        print(f"SE:        {format_score(se)}")
        print(f"RF:        {format_score(rf)}")
        print(f"Spatial:   {format_score(spatial_overall)}")
    print(f"Report:    {report_file}")

    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OmniCoT CoT quality from lmms-eval outputs.")
    parser.add_argument("--input-file", required=True, help="Path to omnicot_submission.json or compatible JSON/JSONL.")
    parser.add_argument("--output-dir", default="results/omnicot_cot_quality", help="Directory for results and checkpoints.")
    parser.add_argument("--mode", choices=("simple", "spatial", "all"), default="all")
    parser.add_argument("--gt-dir", default=None, help="Optional directory containing <scene_id>.txt ground-truth descriptions.")
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--judge-model", default=None, help="Override JUDGE_MODEL for this run.")
    parser.add_argument("--base-url", default=None, help="Override OPENAI_BASE_URL for this run.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.judge_model:
        os.environ["JUDGE_MODEL"] = args.judge_model
    if args.base_url:
        os.environ["OPENAI_BASE_URL"] = args.base_url

    input_path = Path(args.input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_file = output_dir / "cot_quality_results.json"
    checkpoint_file = output_dir / "cot_quality_results.jsonl"

    raw_items = load_input(input_path)
    if args.max_samples is not None:
        raw_items = raw_items[: args.max_samples]
    items = [normalize_item(item, index, args.gt_dir) for index, item in enumerate(raw_items)]

    reset_checkpoint = os.environ.get("RESET_CHECKPOINT", "0") == "1"
    if reset_checkpoint and checkpoint_file.exists():
        checkpoint_file.unlink()

    cached = {} if reset_checkpoint else load_cached_results(checkpoint_file, final_file)
    keys = [item_key(item, index) for index, item in enumerate(items)]
    pending = [
        (index, item, keys[index])
        for index, item in enumerate(items)
        if keys[index] not in cached
    ]

    print(f"[INFO] Input: {input_path}")
    print(f"[INFO] Output dir: {output_dir}")
    print(f"[INFO] Mode: {args.mode}")
    print(f"[INFO] Judge model: {os.environ.get('JUDGE_MODEL', DEFAULT_JUDGE_MODEL)}")
    print(f"[INFO] Total samples: {len(items)}")
    print(f"[INFO] Reused samples: {len(items) - len(pending)}")
    print(f"[INFO] Pending samples: {len(pending)}")
    print(f"[INFO] Checkpoint: {checkpoint_file}")

    if pending and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("[ERROR] OPENAI_API_KEY is required because there are pending samples.")

    if pending:
        with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = {
                executor.submit(evaluate_single_item, item, args.mode): (index, item, key)
                for index, item, key in pending
            }
            iterator = as_completed(futures)
            if HAS_TQDM:
                iterator = tqdm(iterator, total=len(futures), desc="CoT quality")

            for future in iterator:
                index, item, key = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        "item_id": item["QA_id"],
                        "QA_id": item["QA_id"],
                        "fatal_error": str(exc),
                    }
                result["__item_key"] = key
                result["__index"] = index
                cached[key] = result
                append_checkpoint(checkpoint_file, result)

    ordered_results = [
        strip_internal_keys(cached[key])
        for key in keys
        if key in cached
    ]
    final_file.write_text(json.dumps(ordered_results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[SUCCESS] Results: {final_file}")
    generate_report(ordered_results, args.mode, output_dir)


if __name__ == "__main__":
    main()
