from openai import OpenAI
from typing import List, Dict, Tuple
import json

from src.prompt_loader import PromptLoader
from src.runtime_config import get_api_settings, get_model_settings


class QuestionGenerator:
    QUESTION_TYPES = [
        'viewpoint_transform_identify',
        'viewpoint_transform_angle',
        'multi_hop_object',
        'multi_hop_direction',
        'move_translation',
        'move_turn_combined',
    ]

    TYPE_DESCRIPTIONS = {
        'viewpoint_transform_identify': {
            'name': 'Viewpoint Transform - Object Identification',
            'structure': 'Standing at [specific object], facing [direction/target], turn [angle1]° [dir1], then turn [angle2]° [dir2], what is the NEAREST object?',
            'key': 'Tests spatial orientation and multi-step rotation from object position',
        },
        'viewpoint_transform_angle': {
            'name': 'Viewpoint Transform - Angle Calculation',
            'structure': 'Standing at [object_A], initially facing [direction/object_B], turn to face [object_C], then turn to face [object_D], what is the TOTAL angle turned? (Choose closest: 45°/90°/135°/180°)',
            'key': 'Tests angle calculation between directional transformations',
        },
        'multi_hop_object': {
            'name': 'Multi-Hop Object Identification',
            'structure': 'What is [relation2] of the [relation1 + qualifier] object of [anchor_object]?',
            'key': 'Tests spatial relation chain reasoning',
        },
        'multi_hop_direction': {
            'name': 'Multi-Hop Direction Identification',
            'structure': 'In which direction is the [relation1 + qualifier] object of [anchor_object], relative to [anchor_object]?',
            'key': 'Tests inverse spatial reasoning and direction calculation',
        },
        'move_translation': {
            'name': 'Pure Translation Simulation',
            'structure': 'From [start_object], walk [direction] for [distance]m, what is the FIRST object you will see?',
            'key': 'Tests dynamic field-of-view prediction with movement',
        },
        'move_turn_combined': {
            'name': 'Combined Move-Turn Simulation',
            'structure': 'From [start_object], walk [direction] to [end], turn [angle]° [dir], is [target_object] still visible?',
            'key': 'Tests compound spatial transformation',
        },
    }

    def __init__(self, config: dict):
        self.config = config
        api_key, base_url = get_api_settings(config)
        model_settings = get_model_settings(config)
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model_settings['reasoning']
        temp_config = config['generation'].get('temperature', {})
        self.temperature = temp_config.get('reasoning', 0.7)
        self.timeout = config['generation'].get('timeout', 180)
        self.stream = config['generation'].get('stream', True)
        max_tokens_config = config['generation'].get('max_tokens', {})
        self.max_tokens = max_tokens_config.get('question_generation', max_tokens_config.get('text', 2048))
        self.enable_thinking = config['generation'].get('enable_thinking', False)
        self.prompt_loader = PromptLoader(config.get('prompts_dir'))
        missing_descriptions = [qtype for qtype in self.QUESTION_TYPES if qtype not in self.TYPE_DESCRIPTIONS]
        if missing_descriptions:
            raise ValueError(f'Missing type descriptions for: {missing_descriptions}')

    def _clean_json_response(self, content: str) -> str:
        if content.startswith('```json'):
            content = content[7:]
        elif content.startswith('```'):
            content = content[3:]
        if content.endswith('```'):
            content = content[:-3]
        return content.strip()

    def _validate_and_fix_json(self, content: str) -> Dict:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            content = content.replace(',\n  ]\n}', '\n  ]\n}')
            content = content.replace(',\n]', '\n]')
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {'questions': []}

    def _load_type_guidance(self, question_type: str) -> str:
        path = f'question_generator/type_guidance/{question_type}.txt'
        try:
            return self.prompt_loader.load(path).strip()
        except FileNotFoundError:
            return ''

    def _build_specific_schema_item(self, question_type: str) -> str:
        if question_type == 'viewpoint_transform_angle':
            return self.prompt_loader.render('question_generator/specific_item_angle.txt', question_type=question_type).rstrip()
        return self.prompt_loader.render('question_generator/specific_item_default.txt', question_type=question_type).rstrip()

    def _collect_response(self, response) -> Tuple[str, Dict]:
        usage_info = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        if self.stream:
            content = ''
            final_usage = None
            for chunk in response:
                if hasattr(chunk, 'usage') and chunk.usage:
                    final_usage = {
                        'prompt_tokens': chunk.usage.prompt_tokens,
                        'completion_tokens': chunk.usage.completion_tokens,
                        'total_tokens': chunk.usage.total_tokens
                    }
                if chunk.choices and hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content:
                    content += chunk.choices[0].delta.content
            if final_usage:
                usage_info = final_usage
            return content, usage_info

        content = response.choices[0].message.content
        if hasattr(response, 'usage'):
            usage_info = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
        return content, usage_info

    def generate_specific_type_question(self, scene_description: str, question_type: str, count: int = 1) -> Tuple[List[Dict], Dict]:
        if question_type not in self.TYPE_DESCRIPTIONS:
            raise ValueError(f'Unknown question type: {question_type}. Valid types: {list(self.TYPE_DESCRIPTIONS.keys())}')

        type_info = self.TYPE_DESCRIPTIONS[question_type]
        type_guidance = self._load_type_guidance(question_type)

        prompt = self.prompt_loader.render(
            'question_generator/specific_base.txt',
            count=count,
            scene_description=scene_description,
            question_type=question_type,
            type_name=type_info['name'],
            type_structure=type_info['structure'],
            type_key=type_info['key'],
            type_guidance=type_guidance
        )

        prompt += '\n\n' + self.prompt_loader.load('question_generator/specific_output_header.txt').rstrip() + '\n'
        item_blocks = [self._build_specific_schema_item(question_type) for _ in range(count)]
        prompt += ',\n'.join(item_blocks) + '\n'
        prompt += self.prompt_loader.load('question_generator/specific_output_footer.txt').rstrip() + '\n\n'
        prompt += self.prompt_loader.render(
            'question_generator/specific_tail.txt',
            count=count,
            question_type=question_type
        ).strip()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    'role': 'system',
                    'content': self.prompt_loader.render('question_generator/system_specific.txt', type_name=type_info['name']).strip()
                },
                {'role': 'user', 'content': prompt}
            ],
            temperature=self.temperature,
            timeout=self.timeout,
            stream=self.stream,
            max_tokens=self.max_tokens,
            stream_options={'include_usage': True} if self.stream else None,
            extra_body={'enable_thinking': self.enable_thinking}
        )

        content, usage_info = self._collect_response(response)
        cleaned_content = self._clean_json_response(content)
        result = self._validate_and_fix_json(cleaned_content)
        questions = result.get('questions', [])

        for question in questions:
            if question.get('type') != question_type:
                question['type'] = question_type

        return questions, usage_info

    def generate_questions(self, scene_description: str, num_questions: int = 6) -> Tuple[List[Dict], Dict]:
        prompt = self.prompt_loader.render('question_generator/generate_all.txt', scene_description=scene_description)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {'role': 'system', 'content': self.prompt_loader.load('question_generator/system_general.txt').strip()},
                {'role': 'user', 'content': prompt}
            ],
            temperature=self.temperature,
            timeout=self.timeout,
            stream=self.stream,
            max_tokens=self.max_tokens,
            stream_options={'include_usage': True} if self.stream else None,
            extra_body={'enable_thinking': self.enable_thinking}
        )

        content, usage_info = self._collect_response(response)
        cleaned_content = self._clean_json_response(content)
        result = self._validate_and_fix_json(cleaned_content)
        questions = result.get('questions', [])
        if num_questions > 0:
            questions = questions[:num_questions]
        return questions, usage_info

    def generate_questions_batch(self, scene_description: str, batch_count: int = 1) -> Tuple[List[Dict], Dict]:
        all_questions = []
        total_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

        questions_per_batch = len(self.QUESTION_TYPES)
        print(f'   Generating {batch_count} batches (each batch = {questions_per_batch} questions, total = {batch_count * questions_per_batch})...')

        for batch_idx in range(batch_count):
            print(f'     Batch {batch_idx + 1}/{batch_count}...', end=' ')
            questions, usage = self.generate_questions(scene_description, num_questions=questions_per_batch)
            for question in questions:
                question['batch_id'] = batch_idx
                question['generation_batch_count'] = batch_count
            all_questions.extend(questions)
            total_usage['prompt_tokens'] += usage['prompt_tokens'] or 0
            total_usage['completion_tokens'] += usage['completion_tokens']
            total_usage['total_tokens'] += usage['total_tokens']
            print(f'Got {len(questions)} questions ({usage["total_tokens"]} tokens)')

        type_counts = {}
        for question in all_questions:
            question_type = question.get('type', 'unknown')
            type_counts[question_type] = type_counts.get(question_type, 0) + 1

        print(f'\n   Total generated: {len(all_questions)} questions')
        print('    Type distribution:')
        for question_type, count in sorted(type_counts.items()):
            print(f'      {question_type}: {count} questions')

        return all_questions, total_usage

