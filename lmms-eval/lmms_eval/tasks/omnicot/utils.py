import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from PIL import Image


CURRENT_DIR = Path(__file__).parent
DATA_DIR = Path(os.environ.get("OMNICOT_DATA_DIR", str(CURRENT_DIR / "sample_data")))
IMAGE_DIR = Path(os.environ.get("OMNICOT_IMAGE_DIR", str(DATA_DIR / "image")))


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def _existing_path(candidates: Iterable[Path]) -> Optional[Path]:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _candidate_paths(image_path: str) -> List[Path]:
    path = Path(str(image_path))
    basename = path.name
    normalized = str(image_path).replace("\\", "/")
    candidates = [
        path,
        IMAGE_DIR / basename,
        DATA_DIR / normalized,
        DATA_DIR / basename,
        DATA_DIR / "image" / basename,
    ]
    return candidates


def _load_image(path: Path):
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def omnicot_doc_to_visual(doc: Dict) -> List:
    images = []
    image_items = _as_list(doc.get("image"))

    for item in image_items:
        real_path = _existing_path(_candidate_paths(str(item)))
        if real_path is None:
            continue
        image = _load_image(real_path)
        if image is not None:
            images.append(image)

    if images:
        return images

    return []


def omnicot_doc_to_visual_empty(doc: Dict) -> List:
    return []


def _visual_hint(doc: Dict) -> str:
    image_items = _as_list(doc.get("image"))
    if len(image_items) == 1:
        return "Visual Input: The input contains one ERP panoramic image.\n"

    return "Visual Input: Use the provided panoramic image when reasoning.\n"


def _reference_objects(doc: Dict, limit: int = 3) -> str:
    objects = _as_list(doc.get("random_objects"))[:limit]
    if not objects:
        return ""
    lines = "\n".join(str(item) for item in objects)
    return (
        "Reference object locations, randomly selected and not exhaustive:\n"
        f"{lines}\n"
    )


def _format_prompt(doc: Dict, lmms_eval_specific_kwargs: Dict, include_desc: bool, include_visual_hint: bool) -> str:
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = doc.get("question", "")
    parts = [pre_prompt.strip()] if pre_prompt else []

    if include_visual_hint:
        parts.append(_visual_hint(doc).strip())

    parts.append("Coordinate System: X-axis points east. Y-axis points north.")

    reference = _reference_objects(doc, limit=10 if include_desc else 3)
    if reference:
        parts.append(reference.strip())

    if include_desc and doc.get("description"):
        parts.append(f"Scene Description:\n{doc.get('description')}")

    parts.append(f"Question: {question}")

    if post_prompt:
        parts.append(f"Requirements: {post_prompt}")

    return "\n\n".join(part for part in parts if part)


def omnicot_doc_to_text_no_desc(doc: Dict, lmms_eval_specific_kwargs: Dict) -> str:
    return _format_prompt(doc, lmms_eval_specific_kwargs, include_desc=False, include_visual_hint=True)


def omnicot_doc_to_text_with_desc(doc: Dict, lmms_eval_specific_kwargs: Dict) -> str:
    return _format_prompt(doc, lmms_eval_specific_kwargs, include_desc=True, include_visual_hint=True)


def omnicot_doc_to_text_text_only(doc: Dict, lmms_eval_specific_kwargs: Dict) -> str:
    return _format_prompt(doc, lmms_eval_specific_kwargs, include_desc=False, include_visual_hint=False)


def _messages(text: str, images: List) -> List[Dict]:
    content = [{"type": "text", "text": text}]
    for image in images:
        content.append({"type": "image", "url": image})
    return [{"role": "user", "content": content}]


def omnicot_doc_to_messages_no_desc(doc: Dict, lmms_eval_specific_kwargs: Dict) -> List[Dict]:
    return _messages(
        omnicot_doc_to_text_no_desc(doc, lmms_eval_specific_kwargs),
        omnicot_doc_to_visual(doc),
    )


def omnicot_doc_to_messages_with_desc(doc: Dict, lmms_eval_specific_kwargs: Dict) -> List[Dict]:
    return _messages(
        omnicot_doc_to_text_with_desc(doc, lmms_eval_specific_kwargs),
        omnicot_doc_to_visual(doc),
    )


def omnicot_doc_to_messages_text_only(doc: Dict, lmms_eval_specific_kwargs: Dict) -> List[Dict]:
    return _messages(
        omnicot_doc_to_text_text_only(doc, lmms_eval_specific_kwargs),
        [],
    )


def extract_answer(response: str) -> str:
    response = str(response or "").strip()
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    answer_match = re.search(r"(?:final\s+)?answer\s*:\s*(.+)$", response, re.IGNORECASE | re.DOTALL)
    if answer_match:
        return answer_match.group(1).strip().splitlines()[0].strip()

    lines = [line.strip() for line in response.splitlines() if line.strip()]
    return lines[-1] if lines else "Failed to Answer"


def extract_reasoning(response: str) -> str:
    response = str(response or "").strip()
    match = re.search(r"<think>\s*(.*?)\s*</think>", response, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()

    parts = re.split(r"<answer>", response, flags=re.IGNORECASE)
    if len(parts) > 1:
        return re.sub(r"</think>\s*$", "", parts[0].strip(), flags=re.IGNORECASE)

    return ""


def normalize_answer(answer: str) -> str:
    text = str(answer or "").lower().strip()
    text = re.sub(r"^[a-d]\s*[\).:-]\s*", "", text)
    text = re.sub(r"^['\"]|['\"]$", "", text)

    number_match = re.search(r"\b(\d{1,3})\b", text)
    if number_match:
        return number_match.group(1)

    text = text.replace("_", " ")
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    text = re.sub(r"[^a-z0-9]", "", text)
    return text


def _contains_match(pred: str, target: str) -> bool:
    if pred == target:
        return True

    directions = {
        "north", "south", "east", "west",
        "northeast", "northwest", "southeast", "southwest",
    }
    if pred in directions or target in directions:
        return False

    shorter, longer = (pred, target) if len(pred) <= len(target) else (target, pred)
    return bool(shorter and shorter in longer)


def _task_group(question_type: str) -> str:
    if question_type in {"viewpoint_transform_identify", "viewpoint_transform_angle"}:
        return "See"
    if question_type.startswith("multi_hop"):
        return "Locate"
    if question_type.startswith("move"):
        return "Move"
    return ""


def omnicot_process_results(doc: Dict, result: Any) -> Dict:
    if isinstance(result, list):
        result = result[0] if result else ""

    prediction = str(result or "").strip()
    pred = extract_answer(prediction)
    reasoning = extract_reasoning(prediction)
    target = doc.get("answer", "")

    pred_normalized = normalize_answer(pred)
    target_normalized = normalize_answer(target)
    exact_match = pred_normalized == target_normalized
    contains_match = _contains_match(pred_normalized, target_normalized)

    question_type = doc.get("type", "")
    group = _task_group(question_type)

    by_group = {
        "accuracy_See": exact_match if group == "See" else None,
        "accuracy_Locate": exact_match if group == "Locate" else None,
        "accuracy_Move": exact_match if group == "Move" else None,
        "contains_accuracy_See": contains_match if group == "See" else None,
        "contains_accuracy_Locate": contains_match if group == "Locate" else None,
        "contains_accuracy_Move": contains_match if group == "Move" else None,
    }

    return {
        "accuracy": exact_match,
        "contains_accuracy": contains_match,
        **by_group,
        "submission": {
            "QA_id": doc.get("QA_id", ""),
            "scene_id": doc.get("scene_id", ""),
            "question": doc.get("question", ""),
            "answer": target,
            "prediction": prediction,
            "pred_extracted": pred,
            "reasoning": reasoning,
            "reference_cot": doc.get("CoT", doc.get("reference_cot", "")),
            "type": question_type,
            "subtype": doc.get("subtype", ""),
            "is_correct": exact_match,
            "is_contains_correct": contains_match,
        },
    }


def _mean_bool(results: List) -> float:
    valid = [item for item in results if item is not None]
    if not valid:
        return 0.0
    return sum(1 for item in valid if item) / len(valid)


def omnicot_aggregate_accuracy(results: List) -> float:
    return _mean_bool(results)


def omnicot_aggregate_contains_accuracy(results: List) -> float:
    return _mean_bool(results)


def omnicot_aggregate_accuracy_See(results: List) -> float:
    return _mean_bool(results)


def omnicot_aggregate_accuracy_Locate(results: List) -> float:
    return _mean_bool(results)


def omnicot_aggregate_accuracy_Move(results: List) -> float:
    return _mean_bool(results)


def omnicot_aggregate_contains_accuracy_See(results: List) -> float:
    return _mean_bool(results)


def omnicot_aggregate_contains_accuracy_Locate(results: List) -> float:
    return _mean_bool(results)


def omnicot_aggregate_contains_accuracy_Move(results: List) -> float:
    return _mean_bool(results)


def omnicot_aggregate_submission(results: List[Dict], args) -> None:
    from lmms_eval.tasks._task_utils.file_utils import generate_submission_file

    path = generate_submission_file("omnicot_submission.json", args)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)
