from openai import OpenAI
import json
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from src.prompt_loader import PromptLoader
from src.runtime_config import get_api_settings, get_model_settings


class QuestionScorer:
    def __init__(self, config: dict):
        api_key, base_url = get_api_settings(config)
        model_settings = get_model_settings(config)
        self.client_qwen = OpenAI(api_key=api_key, base_url=base_url)
        self.client_deepseek = OpenAI(api_key=api_key, base_url=base_url)
        self.models = {
            'deepseek': model_settings['reasoning'],
            'qwen': model_settings['text']
        }
        self.temperature = config['generation']['temperature'].get('scoring', 0.2)
        self.timeout = config['generation']['timeout']
        self.max_tokens = config['generation']['max_tokens'].get('question_scoring', 512)
        self.max_retries = config['generation'].get('max_retries', 3)
        cache_root = Path(config.get('cache_dir', 'data/cache'))
        self.cache_dir = cache_root / 'question_scores'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_cache = True
        self.prompt_loader = PromptLoader(config.get('prompts_dir'))
        self.type_mapping = {
            'viewpoint_transform_identify': 'viewpoint_transform',
            'viewpoint_transform_angle': 'viewpoint_transform',
            'multi_hop_object': 'multi_step_viewpoint',
            'multi_hop_direction': 'multi_step_viewpoint',
            'move_translation': 'move_translation',
            'move_turn_combined': 'move_translation',
        }

    def _get_cache_key(self, scene_description: str, question: Dict, model_name: str) -> str:
        content = f"{scene_description[:500]}|{question.get('question', '')}|{model_name}"
        return hashlib.md5(content.encode()).hexdigest()

    def _load_from_cache(self, cache_key: str) -> Dict:
        if not self.enable_cache:
            return None
        cache_file = self.cache_dir / f'{cache_key}.json'
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as file:
                    return json.load(file)
            except Exception:
                return None
        return None

    def _save_to_cache(self, cache_key: str, data: Dict):
        if not self.enable_cache:
            return
        cache_file = self.cache_dir / f'{cache_key}.json'
        try:
            with open(cache_file, 'w', encoding='utf-8') as file:
                json.dump(data, file, ensure_ascii=False, indent=2)
        except Exception:
            return

    def _get_type_specific_guidance(self, question: Dict) -> str:
        original_type = question.get('type', 'viewpoint_transform_identify')
        mapped_type = self.type_mapping.get(original_type, 'unknown')
        file_name = f'question_scorer/type_guidance/{mapped_type}.txt'
        try:
            return self.prompt_loader.load(file_name).strip()
        except FileNotFoundError:
            return self.prompt_loader.load('question_scorer/type_guidance/unknown.txt').strip()

    def _contains_coordinate(self, text: str) -> bool:
        pure_coord_pattern = r'\(\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*\)'
        object_coord_pattern = r'\([A-Za-z_]+_-?\d+(?:\.\d+)?_-?\d+(?:\.\d+)?\)'
        return bool(re.search(pure_coord_pattern, text) or re.search(object_coord_pattern, text))

    def _build_prompt(self, scene_description: str, question: Dict) -> str:
        original_type = question.get('type', 'viewpoint_transform_identify')
        mapped_type = self.type_mapping.get(original_type, 'viewpoint_transform')
        type_guidance = self._get_type_specific_guidance(question)
        return self.prompt_loader.render(
            'question_scorer/evaluate_question.txt',
            scene_description=scene_description,
            question_text=question.get('question', ''),
            original_type=original_type,
            mapped_type=mapped_type,
            expected_reasoning_steps=question.get('expected_reasoning_steps', 2),
            type_guidance=type_guidance
        )

    def _score_with_single_model(self, scene_description: str, question: Dict, model_name: str) -> Tuple[int, Dict, Dict]:
        cache_key = self._get_cache_key(scene_description, question, model_name)
        cached = self._load_from_cache(cache_key)
        if cached:
            details = {
                'score': cached['score'],
                'tokens': cached['usage']['total_tokens'],
                'prompt_tokens': cached['usage']['prompt_tokens'],
                'completion_tokens': cached['usage']['completion_tokens'],
                'duration': 0,
                'from_cache': True,
                'failed': False
            }
            return cached['score'], cached['usage'], details

        client = self.client_deepseek if model_name == 'deepseek' else self.client_qwen
        model = self.models[model_name]
        prompt = self._build_prompt(scene_description, question)

        import time
        start_time = time.time()

        for attempt in range(self.max_retries):
            try:
                params = {
                    'model': model,
                    'messages': [
                        {'role': 'system', 'content': self.prompt_loader.load('question_scorer/system.txt').strip()},
                        {'role': 'user', 'content': prompt}
                    ],
                    'temperature': self.temperature,
                    'timeout': self.timeout,
                    'max_tokens': self.max_tokens
                }
                if model_name == 'deepseek':
                    params['extra_body'] = {'enable_thinking': False}

                response = client.chat.completions.create(**params)
                duration = time.time() - start_time
                raw_content = response.choices[0].message.content.strip()
                raw_content = re.sub(r'```json\s*', '', raw_content)
                raw_content = re.sub(r'```\s*', '', raw_content)
                match = re.search(r'\{[^{}]*\}', raw_content)
                if match:
                    raw_content = match.group()

                result = json.loads(raw_content)
                score = int(result.get('score', 5))
                score = max(0, min(10, score))

                usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
                if hasattr(response, 'usage') and response.usage:
                    usage = {
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens
                    }

                details = {
                    'score': score,
                    'tokens': usage['total_tokens'],
                    'prompt_tokens': usage['prompt_tokens'],
                    'completion_tokens': usage['completion_tokens'],
                    'duration': duration,
                    'from_cache': False,
                    'failed': False
                }

                self._save_to_cache(
                    cache_key,
                    {
                        'score': score,
                        'usage': usage,
                        'timestamp': datetime.now().isoformat(),
                        'model': model_name,
                        'question_type': question.get('type', 'unknown')
                    }
                )
                return score, usage, details
            except Exception as error:
                if attempt < self.max_retries - 1:
                    time.sleep(5 * (attempt + 1))
                    continue
                raise Exception(f'[{model_name}] Failed after {self.max_retries} attempts: {error}')

        raise Exception(f'[{model_name}] Failed to score question')

    def _score_single_question(self, scene_description: str, question: Dict) -> Tuple[float, Dict, Dict]:
        question_text = question.get('question', '')
        if self._contains_coordinate(question_text):
            model_details = {
                'deepseek': {
                    'score': 0,
                    'tokens': 0,
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'duration': 0,
                    'from_cache': False,
                    'failed': False
                },
                'qwen': {
                    'score': 0,
                    'tokens': 0,
                    'prompt_tokens': 0,
                    'completion_tokens': 0,
                    'duration': 0,
                    'from_cache': False,
                    'failed': False
                }
            }
            usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
            return 0.0, usage, model_details

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_deepseek = executor.submit(self._score_with_single_model, scene_description, question, 'deepseek')
            future_qwen = executor.submit(self._score_with_single_model, scene_description, question, 'qwen')
            score_deepseek, usage_deepseek, details_deepseek = future_deepseek.result()
            score_qwen, usage_qwen, details_qwen = future_qwen.result()

        avg_score = (score_deepseek + score_qwen) / 2.0
        total_usage = {
            'prompt_tokens': usage_deepseek['prompt_tokens'] + usage_qwen['prompt_tokens'],
            'completion_tokens': usage_deepseek['completion_tokens'] + usage_qwen['completion_tokens'],
            'total_tokens': usage_deepseek['total_tokens'] + usage_qwen['total_tokens']
        }
        model_details = {'deepseek': details_deepseek, 'qwen': details_qwen}
        return avg_score, total_usage, model_details

    def score_questions(self, scene_description: str, questions: List[Dict]) -> Tuple[List[Dict], Dict]:
        scored_questions = []
        total_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_idx = {
                executor.submit(self._score_single_question, scene_description, question): (idx, question)
                for idx, question in enumerate(questions)
            }

            for future in as_completed(future_to_idx):
                idx, question = future_to_idx[future]
                try:
                    avg_score, usage, model_details = future.result()
                    question_copy = question.copy()
                    question_copy['score'] = avg_score
                    question_copy['recommendation'] = 'ACCEPT' if avg_score >= 7.0 else ('REVISE' if avg_score >= 5.0 else 'REJECT')
                    question_copy['model_details'] = model_details
                    scored_questions.append(question_copy)
                    total_usage['prompt_tokens'] += usage['prompt_tokens']
                    total_usage['completion_tokens'] += usage['completion_tokens']
                    total_usage['total_tokens'] += usage['total_tokens']
                except Exception as error:
                    question_copy = question.copy()
                    question_copy['score'] = 0
                    question_copy['recommendation'] = 'REJECT'
                    question_copy['model_details'] = {
                        'deepseek': {'failed': True, 'error': str(error)},
                        'qwen': {'failed': True, 'error': str(error)}
                    }
                    scored_questions.append(question_copy)
                    print(f'    Question {idx + 1} failed: {str(error)[:80]}')

        scored_questions.sort(key=lambda entry: entry.get('score', 0), reverse=True)
        return scored_questions, total_usage

    def select_top_questions(self, scored_questions: List[Dict], n: int = 1, min_score: float = 5.0) -> List[Dict]:
        acceptable = [question for question in scored_questions if question.get('score', 0) >= min_score]
        return acceptable[:n]

    def select_top_questions_by_type(self, scored_questions: List[Dict], top_k_per_type: int = 3, min_score: float = 5.0) -> Tuple[List[Dict], List[Dict]]:
        acceptable = [question for question in scored_questions if question.get('score', 0) >= min_score]
        grouped = {}
        for question in acceptable:
            question_type = question.get('type', 'unknown')
            grouped.setdefault(question_type, []).append(question)

        active_questions = []
        reserved_questions = []

        for question_type, items in grouped.items():
            sorted_items = sorted(items, key=lambda entry: entry.get('score', 0), reverse=True)
            active_questions.extend(sorted_items[:top_k_per_type])
            reserved_questions.extend(sorted_items[top_k_per_type:])

        return active_questions, reserved_questions

