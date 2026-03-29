from openai import OpenAI
from typing import Tuple, Dict
import re

from src.api_retry import api_retry_with_backoff
from src.prompt_loader import PromptLoader
from src.runtime_config import get_api_settings, get_model_settings


def clean_text(text: str) -> str:
    if not text:
        return text
    text = re.sub(r'[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]', '', text)
    text = text.encode('utf-8', errors='ignore').decode('utf-8')
    return text


class CoTGenerator:
    def __init__(self, config: dict):
        self.config = config
        api_key, base_url = get_api_settings(config)
        model_settings = get_model_settings(config)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.reasoning_model = model_settings['reasoning']
        self.summary_model = model_settings['reasoning']
        self.reasoning_temp = config['generation']['temperature']['reasoning']
        self.summary_temp = config['generation']['temperature']['summarization']
        self.max_tokens_reasoning = config['generation']['max_tokens']['reasoning']
        self.max_tokens_text = config['generation']['max_tokens']['text']
        self.timeout = config['generation']['timeout']
        self.stream = config['generation']['stream']
        self.enable_thinking = config['generation']['enable_thinking']
        self.prompt_loader = PromptLoader(config.get('prompts_dir'))
        self.type_mapping = {
            'viewpoint_transform_identify': 'viewpoint_transform',
            'viewpoint_transform_angle': 'viewpoint_transform',
            'multi_hop_object': 'multi_step_viewpoint',
            'multi_hop_direction': 'multi_step_viewpoint',
            'move_translation': 'move_translation',
            'move_turn_combined': 'move_translation',
        }

    def _mapped_type(self, question_type: str) -> str:
        return self.type_mapping.get(question_type, 'viewpoint_transform')

    def _load_reasoning_guidance(self, mapped_type: str) -> str:
        return self.prompt_loader.load(f'cot_generator/reasoning_type/{mapped_type}.txt').strip()

    def _load_summarize_guidance(self, mapped_type: str) -> str:
        return self.prompt_loader.load(f'cot_generator/summarize_type/{mapped_type}.txt').strip()

    def _build_reasoning_prompt(self, scene_description: str, question: str, question_type: str) -> str:
        mapped_type = self._mapped_type(question_type)
        type_guidance = self._load_reasoning_guidance(mapped_type)
        return self.prompt_loader.render(
            'cot_generator/reasoning_main.txt',
            scene_description=scene_description,
            question=question,
            original_type=question_type,
            mapped_type=mapped_type,
            type_guidance=type_guidance
        )

    @api_retry_with_backoff()
    def generate_reasoning(self, scene_description: str, question: str, question_metadata: Dict = None) -> Tuple[str, Dict]:
        original_type = 'viewpoint_transform_identify'
        if question_metadata:
            original_type = question_metadata.get('type', original_type)

        prompt = self._build_reasoning_prompt(scene_description, question, original_type)
        response = self.client.chat.completions.create(
            model=self.reasoning_model,
            messages=[
                {
                    'role': 'system',
                    'content': self.prompt_loader.load('cot_generator/reasoning_system.txt').strip()
                },
                {'role': 'user', 'content': prompt}
            ],
            temperature=self.reasoning_temp,
            timeout=self.timeout,
            stream=self.stream,
            max_tokens=self.max_tokens_reasoning,
            stream_options={'include_usage': True} if self.stream else None,
            extra_body={'enable_thinking': self.enable_thinking}
        )

        usage_info = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        reasoning_content = ''
        answer_content = ''
        final_usage = None

        if self.stream:
            for chunk in response:
                if hasattr(chunk, 'usage') and chunk.usage:
                    final_usage = {
                        'prompt_tokens': chunk.usage.prompt_tokens,
                        'completion_tokens': chunk.usage.completion_tokens,
                        'total_tokens': chunk.usage.total_tokens
                    }

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    reasoning_content += delta.reasoning_content
                if hasattr(delta, 'content') and delta.content:
                    answer_content += delta.content

            if final_usage:
                usage_info = final_usage
            else:
                estimated_completion = (len(reasoning_content) + len(answer_content)) // 4
                estimated_prompt = len(prompt) // 4
                usage_info = {
                    'prompt_tokens': estimated_prompt,
                    'completion_tokens': estimated_completion,
                    'total_tokens': estimated_prompt + estimated_completion
                }
        else:
            answer_content = response.choices[0].message.content
            if hasattr(response, 'usage') and response.usage:
                usage_info = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }

        reasoning_content = clean_text(reasoning_content)
        answer_content = clean_text(answer_content)
        full_reasoning = reasoning_content if (self.enable_thinking and reasoning_content) else answer_content
        if reasoning_content and answer_content:
            full_reasoning = reasoning_content + '\n\n' + answer_content

        return full_reasoning, usage_info

    @api_retry_with_backoff()
    def summarize_answer(self, reasoning: str, question: str, question_type: str = 'viewpoint_transform_identify') -> Tuple[str, Dict]:
        mapped_type = self._mapped_type(question_type)
        type_guidance = self._load_summarize_guidance(mapped_type)

        prompt = self.prompt_loader.render(
            'cot_generator/summarize_main.txt',
            question=question,
            original_type=question_type,
            mapped_type=mapped_type,
            reasoning=reasoning,
            type_guidance=type_guidance
        )

        response = self.client.chat.completions.create(
            model=self.summary_model,
            messages=[
                {
                    'role': 'system',
                    'content': self.prompt_loader.load('cot_generator/summarize_system.txt').strip()
                },
                {'role': 'user', 'content': prompt}
            ],
            temperature=self.summary_temp,
            timeout=self.timeout,
            max_tokens=self.max_tokens_text
        )

        answer = clean_text(response.choices[0].message.content.strip())
        usage_info = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        if hasattr(response, 'usage') and response.usage:
            usage_info = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }

        return answer, usage_info

