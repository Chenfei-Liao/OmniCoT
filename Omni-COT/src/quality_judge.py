from openai import OpenAI
import json
import re
import hashlib
from pathlib import Path
from typing import Dict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from src.prompt_loader import PromptLoader
from src.runtime_config import get_api_settings, get_model_settings


class QualityJudge:
    def __init__(self, config: dict):
        api_key, base_url = get_api_settings(config)
        model_settings = get_model_settings(config)
        self.client_qwen = OpenAI(api_key=api_key, base_url=base_url)
        self.client_deepseek = OpenAI(api_key=api_key, base_url=base_url)
        self.models = {
            'deepseek': model_settings['reasoning'],
            'qwen': model_settings['text']
        }
        self.temperature = config['generation']['temperature'].get('evaluation', 0.1)
        self.timeout = config['generation']['timeout']
        self.max_tokens = config['generation']['max_tokens'].get('quality_evaluation', 1024)
        self.max_retries = config['generation'].get('max_retries', 3)
        self.prompt_loader = PromptLoader(config.get('prompts_dir'))

        cache_root = Path(config.get('cache_dir', 'data/cache'))
        self.cache_dir = cache_root / 'quality_evaluations'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enable_cache = True

        self.type_mapping = {
            'viewpoint_transform_identify': 'viewpoint_transform',
            'viewpoint_transform_angle': 'viewpoint_transform',
            'multi_hop_object': 'multi_step_viewpoint',
            'multi_hop_direction': 'multi_step_viewpoint',
            'move_translation': 'move_translation',
            'move_turn_combined': 'move_translation',
        }

    def _get_cache_key(self, scene_description: str, question: str, cot_reasoning: str, final_answer: str, question_type: str, model_name: str) -> str:
        content = f"{scene_description[:300]}|{question[:200]}|{cot_reasoning[:500]}|{final_answer[:200]}|{question_type}|{model_name}"
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

    def _build_evaluation_prompt(self, scene_description: str, question: str, cot_reasoning: str, final_answer: str, cot: list, original_type: str, mapped_type: str) -> str:
        cot_lines = ''
        if cot:
            cot_lines = '\n'.join([f'{index + 1}. {step}' for index, step in enumerate(cot)])

        return self.prompt_loader.render(
            'quality_judge/evaluate.txt',
            scene_description=scene_description,
            question=question,
            cot_reasoning=cot_reasoning,
            final_answer=final_answer,
            cot_lines=cot_lines,
            original_type=original_type,
            mapped_type=mapped_type
        )

    def _evaluate_with_single_model(self, scene_description: str, question: str, cot_reasoning: str, final_answer: str, cot: list, original_type: str, mapped_type: str, model_name: str) -> tuple:
        cache_key = self._get_cache_key(scene_description, question, cot_reasoning, final_answer, original_type, model_name)
        cached = self._load_from_cache(cache_key)
        if cached:
            return cached['reasoning_score'], cached['answer_score'], cached['usage']

        client = self.client_deepseek if model_name == 'deepseek' else self.client_qwen
        model = self.models[model_name]
        prompt = self._build_evaluation_prompt(scene_description, question, cot_reasoning, final_answer, cot or [], original_type, mapped_type)

        for attempt in range(self.max_retries):
            try:
                params = {
                    'model': model,
                    'messages': [
                        {'role': 'system', 'content': self.prompt_loader.load('quality_judge/system.txt').strip()},
                        {'role': 'user', 'content': prompt}
                    ],
                    'temperature': self.temperature,
                    'timeout': self.timeout,
                    'max_tokens': self.max_tokens
                }
                if model_name == 'deepseek':
                    params['extra_body'] = {'enable_thinking': False}

                response = client.chat.completions.create(**params)
                raw_content = response.choices[0].message.content.strip()
                raw_content = re.sub(r'```json\s*', '', raw_content)
                raw_content = re.sub(r'```\s*', '', raw_content)
                json_match = re.search(r'\{[^{}]*\}', raw_content)
                if json_match:
                    raw_content = json_match.group()

                result = json.loads(raw_content)
                reasoning_score = int(result.get('reasoning_score', 5))
                answer_score = int(result.get('answer_score', 5))
                reasoning_score = max(0, min(10, reasoning_score))
                answer_score = max(0, min(10, answer_score))

                usage_info = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
                if hasattr(response, 'usage') and response.usage:
                    usage_info = {
                        'prompt_tokens': response.usage.prompt_tokens,
                        'completion_tokens': response.usage.completion_tokens,
                        'total_tokens': response.usage.total_tokens
                    }

                self._save_to_cache(
                    cache_key,
                    {
                        'reasoning_score': reasoning_score,
                        'answer_score': answer_score,
                        'usage': usage_info,
                        'timestamp': datetime.now().isoformat(),
                        'model': model_name,
                        'original_type': original_type,
                        'mapped_type': mapped_type
                    }
                )

                return reasoning_score, answer_score, usage_info
            except Exception as error:
                if attempt < self.max_retries - 1:
                    import time
                    time.sleep(5 * (attempt + 1))
                    continue
                raise Exception(f'[{model_name}] Evaluation failed after {self.max_retries} attempts: {error}')

        raise Exception(f'[{model_name}] Evaluation failed')

    def evaluate(self, scene_description: str, question: str, cot_reasoning: str, final_answer: str, cot: list = None, question_metadata: Dict = None) -> dict:
        original_type = 'viewpoint_transform_identify'
        if question_metadata:
            original_type = question_metadata.get('type', original_type)
        mapped_type = self.type_mapping.get(original_type, 'viewpoint_transform')

        import time

        def timed_evaluate(model_name):
            start = time.time()
            r_score, a_score, usage = self._evaluate_with_single_model(
                scene_description,
                question,
                cot_reasoning,
                final_answer,
                cot or [],
                original_type,
                mapped_type,
                model_name
            )
            duration = time.time() - start
            return r_score, a_score, usage, duration

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_deepseek = executor.submit(timed_evaluate, 'deepseek')
            future_qwen = executor.submit(timed_evaluate, 'qwen')
            r_score_ds, a_score_ds, usage_ds, duration_ds = future_deepseek.result()
            r_score_qw, a_score_qw, usage_qw, duration_qw = future_qwen.result()

        reasoning_score_avg = (r_score_ds + r_score_qw) / 2.0
        answer_score_avg = (a_score_ds + a_score_qw) / 2.0
        overall_score = (reasoning_score_avg + answer_score_avg) / 2.0

        total_usage = {
            'prompt_tokens': usage_ds['prompt_tokens'] + usage_qw['prompt_tokens'],
            'completion_tokens': usage_ds['completion_tokens'] + usage_qw['completion_tokens'],
            'total_tokens': usage_ds['total_tokens'] + usage_qw['total_tokens']
        }

        return {
            'reasoning_score': reasoning_score_avg,
            'answer_score': answer_score_avg,
            'overall_score': overall_score,
            'passed': overall_score >= 6.0,
            'usage': total_usage,
            'model_details': {
                'deepseek': {
                    'reasoning_score': r_score_ds,
                    'answer_score': a_score_ds,
                    'tokens': usage_ds['total_tokens'],
                    'prompt_tokens': usage_ds['prompt_tokens'],
                    'completion_tokens': usage_ds['completion_tokens'],
                    'duration': duration_ds,
                    'failed': False
                },
                'qwen': {
                    'reasoning_score': r_score_qw,
                    'answer_score': a_score_qw,
                    'tokens': usage_qw['total_tokens'],
                    'prompt_tokens': usage_qw['prompt_tokens'],
                    'completion_tokens': usage_qw['completion_tokens'],
                    'duration': duration_qw,
                    'failed': False
                }
            }
        }

    def _fallback_evaluation(self):
        return {
            'reasoning_score': 0.0,
            'answer_score': 0.0,
            'overall_score': 0.0,
            'passed': False,
            'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
            'model_details': {
                'deepseek': {'failed': True},
                'qwen': {'failed': True}
            }
        }

