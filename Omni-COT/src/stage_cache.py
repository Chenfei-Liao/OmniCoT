import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


class SimplifiedStageCache:
    CACHE_VERSION = '3.2.0'

    def __init__(self, cache_dir: str = 'data/cache/stage_cache'):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        print(f'[SimplifiedCache] Initialized at: {self.cache_dir}')

    def _get_cache_path(self, scene_id: str) -> Path:
        return self.cache_dir / f'{scene_id}.json'

    def _load_cache(self, scene_id: str) -> Optional[Dict]:
        cache_path = self._get_cache_path(scene_id)
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, 'r', encoding='utf-8') as file:
                return json.load(file)
        except Exception:
            return None

    def _save_cache(self, scene_id: str, cache_data: Dict):
        cache_path = self._get_cache_path(scene_id)
        with open(cache_path, 'w', encoding='utf-8') as file:
            json.dump(cache_data, file, ensure_ascii=False, indent=2)
            file.write('\n')

    def _init_cache(self, scene_id: str) -> Dict:
        now = datetime.now().isoformat()
        return {
            'cache_version': self.CACHE_VERSION,
            'scene_id': scene_id,
            'created_at': now,
            'last_updated': now,
            'stage_1': {'status': 'pending'},
            'stage_3': {'status': 'pending'},
            'qa_pairs': {},
        }

    def _hash_question(self, question_text: str) -> str:
        return hashlib.sha256(question_text.encode('utf-8')).hexdigest()

    def get_stage_1(self, scene_id: str) -> Optional[Dict]:
        cache_data = self._load_cache(scene_id)
        if cache_data is None:
            return None
        stage_1 = cache_data.get('stage_1', {})
        if stage_1.get('status') != 'completed':
            return None
        return {
            'description': stage_1.get('description', ''),
            'tokens': stage_1.get('tokens', 0),
            'random_objects': stage_1.get('random_objects', []),
        }

    def save_stage_1(self, scene_id: str, description: str, usage: Dict, random_objects: List[Dict] = None):
        cache_data = self._load_cache(scene_id) or self._init_cache(scene_id)
        cache_data['stage_1'] = {
            'status': 'completed',
            'description': description,
            'tokens': usage.get('total_tokens', 0),
            'random_objects': random_objects or [],
            'timestamp': datetime.now().isoformat(),
        }
        cache_data['last_updated'] = datetime.now().isoformat()
        self._save_cache(scene_id, cache_data)

    def get_stage_3(self, scene_id: str) -> Optional[Dict]:
        cache_data = self._load_cache(scene_id)
        if cache_data is None:
            return None
        stage_3 = cache_data.get('stage_3', {})
        if stage_3.get('status') != 'completed':
            return None
        return stage_3.get('questions_by_type', {})

    def save_stage_3(self, scene_id: str, questions_by_type: Dict):
        cache_data = self._load_cache(scene_id) or self._init_cache(scene_id)
        cache_data['stage_3'] = {
            'status': 'completed',
            'questions_by_type': questions_by_type,
            'timestamp': datetime.now().isoformat(),
        }
        cache_data['last_updated'] = datetime.now().isoformat()
        self._save_cache(scene_id, cache_data)

    def get_qa_pair(self, scene_id: str, question_text: str) -> Optional[Dict]:
        cache_data = self._load_cache(scene_id)
        if cache_data is None:
            return None
        question_hash = self._hash_question(question_text)
        return cache_data.get('qa_pairs', {}).get(question_hash)

    def save_qa_pair(self, scene_id: str, question_text: str, qa_data: Dict):
        cache_data = self._load_cache(scene_id) or self._init_cache(scene_id)
        question_hash = self._hash_question(question_text)
        cached_qa = {
            'scene_id': scene_id,
            'question_hash': question_hash,
            'question_text': question_text[:100],
            'question_data': qa_data.get('question_data', {}),
            'stage_4': {
                'status': 'completed',
                'reasoning_process': qa_data.get('reasoning_process', {}),
                'answer_data': qa_data.get('answer_data', {}),
                'tokens': qa_data.get('token_usage', {}).get('total', 0),
                'timestamp': qa_data.get('timestamp', datetime.now().isoformat()),
            },
            'stage_5': {
                'status': 'completed',
                'evaluation': qa_data.get('quality_evaluation', {}),
                'tokens': 0,
                'timestamp': qa_data.get('timestamp', datetime.now().isoformat()),
            },
        }
        cache_data.setdefault('qa_pairs', {})[question_hash] = cached_qa
        cache_data['last_updated'] = datetime.now().isoformat()
        self._save_cache(scene_id, cache_data)

    def _generate_qa_id(self, scene_id: str, question_hash: str, index: int = 0) -> str:
        return f'{scene_id}-{index:05d}'

    def _generate_image_path(self, scene_id: str) -> str:
        return f'image/{scene_id}.png'

    def _determine_subtype(self, question_type: str, question_text: str = '') -> str:
        mapping = {
            'viewpoint_transform_identify': 'C2',
            'viewpoint_transform_angle': 'C2',
            'multi_hop_object': 'C3',
            'multi_hop_direction': 'C3',
            'move_translation': 'C4',
            'move_turn_combined': 'C4',
        }
        return mapping.get(question_type, 'C1')

    def export_qa_pairs(self, scene_id: str) -> List[Dict]:
        cache_data = self._load_cache(scene_id)
        if cache_data is None:
            return []

        scene_description = ''
        random_objects = []
        stage_1 = cache_data.get('stage_1', {})
        if stage_1.get('status') == 'completed':
            scene_description = stage_1.get('description', '')
            random_objects = stage_1.get('random_objects', [])

        qa_pairs = cache_data.get('qa_pairs', {})
        exported = []
        for index, (question_hash, qa) in enumerate(sorted(qa_pairs.items())):
            stage_5 = qa.get('stage_5', {})
            if stage_5.get('status') != 'completed':
                continue
            evaluation = stage_5.get('evaluation', {})
            if not evaluation.get('passed', False):
                continue
            stage_4 = qa.get('stage_4', {})
            question_data = qa.get('question_data', {})
            answer_data = stage_4.get('answer_data', {})
            question_type = question_data.get('type', 'unknown')
            question_text = question_data.get('question', '')
            cot_list = answer_data.get('cot', [])
            exported.append(
                {
                    'image': self._generate_image_path(scene_id),
                    'QA_id': self._generate_qa_id(scene_id, question_hash, index),
                    'type': question_type,
                    'scene_id': scene_id,
                    'description': scene_description,
                    'subtype': self._determine_subtype(question_type, question_text),
                    'question': question_text,
                    'answer': answer_data.get('final_answer', ''),
                    'CoT': cot_list,
                    'Steps': str(len(cot_list)),
                    'random_objects': random_objects,
                }
            )
        return exported

    def export_all_qa_pairs(self, output_path: str = None) -> List[Dict]:
        all_qa = []
        for scene_id in sorted(self.list_all_scenes()):
            all_qa.extend(self.export_qa_pairs(scene_id))
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as file:
                json.dump(all_qa, file, ensure_ascii=False, indent=2)
        return all_qa

    def get_scene_progress(self, scene_id: str) -> Dict:
        cache_data = self._load_cache(scene_id)
        if cache_data is None:
            return {'stage_1_completed': False, 'stage_3_completed': False, 'qa_count': 0, 'passed_qa_count': 0}
        qa_pairs = cache_data.get('qa_pairs', {})
        passed = sum(
            1
            for qa in qa_pairs.values()
            if qa.get('stage_5', {}).get('status') == 'completed'
            and qa.get('stage_5', {}).get('evaluation', {}).get('passed', False)
        )
        return {
            'stage_1_completed': cache_data.get('stage_1', {}).get('status') == 'completed',
            'stage_3_completed': cache_data.get('stage_3', {}).get('status') == 'completed',
            'qa_count': len(qa_pairs),
            'passed_qa_count': passed,
        }

    def list_all_scenes(self) -> List[str]:
        return [file.stem for file in self.cache_dir.glob('*.json')]

    def get_statistics(self) -> Dict:
        scenes = self.list_all_scenes()
        stats = {'total_scenes': len(scenes), 'stage_1_completed': 0, 'stage_3_completed': 0, 'total_qa_pairs': 0, 'total_passed_qa': 0}
        for scene_id in scenes:
            progress = self.get_scene_progress(scene_id)
            if progress['stage_1_completed']:
                stats['stage_1_completed'] += 1
            if progress['stage_3_completed']:
                stats['stage_3_completed'] += 1
            stats['total_qa_pairs'] += progress['qa_count']
            stats['total_passed_qa'] += progress['passed_qa_count']
        return stats

    def clear_scene_cache(self, scene_id: str):
        cache_path = self._get_cache_path(scene_id)
        if cache_path.exists():
            cache_path.unlink()

    def clear_all_caches(self):
        for file in self.cache_dir.glob('*.json'):
            file.unlink()

    def reset_stage(self, scene_id: str, stage: int):
        cache_data = self._load_cache(scene_id)
        if cache_data is None:
            return
        if stage == 1:
            cache_data['stage_1'] = {'status': 'pending'}
            cache_data['stage_3'] = {'status': 'pending'}
            cache_data['qa_pairs'] = {}
        elif stage == 3:
            cache_data['stage_3'] = {'status': 'pending'}
            cache_data['qa_pairs'] = {}
        elif stage in [4, 5]:
            cache_data['qa_pairs'] = {}
        cache_data['last_updated'] = datetime.now().isoformat()
        self._save_cache(scene_id, cache_data)

    def validate_cache(self, scene_id: str) -> Dict:
        cache_data = self._load_cache(scene_id)
        if cache_data is None:
            return {'valid': False, 'issues': ['Cache file does not exist']}
        required = ['scene_id', 'cache_version', 'stage_1', 'stage_3', 'qa_pairs']
        issues = [f'Missing required field: {field}' for field in required if field not in cache_data]
        return {'valid': len(issues) == 0, 'issues': issues}

    def compact_cache(self, scene_id: str):
        cache_data = self._load_cache(scene_id)
        if cache_data is None:
            return
        qa_pairs = cache_data.get('qa_pairs', {})
        passed_pairs = {
            key: value
            for key, value in qa_pairs.items()
            if value.get('stage_5', {}).get('evaluation', {}).get('passed', False)
        }
        cache_data['qa_pairs'] = passed_pairs
        cache_data['last_updated'] = datetime.now().isoformat()
        self._save_cache(scene_id, cache_data)

    def view_cache_file(self, scene_id: str, max_lines: int = 50):
        cache_path = self._get_cache_path(scene_id)
        if not cache_path.exists():
            print(f'[Cache] No cache file found for {scene_id}')
            return
        with open(cache_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for index, line in enumerate(lines[:max_lines], start=1):
            print(f'{index:4d} | {line}', end='')


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Simplified Stage Cache Management Tool')
    parser.add_argument('--cache-dir', type=str, default='data/cache/stage_cache')
    parser.add_argument('--stats', action='store_true')
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--scene', type=str)
    parser.add_argument('--view', type=str)
    parser.add_argument('--validate', type=str)
    parser.add_argument('--clear', type=str)
    parser.add_argument('--clear-all', action='store_true')
    parser.add_argument('--compact', type=str)
    parser.add_argument('--export', type=str)
    parser.add_argument('--export-scene', type=str)
    parser.add_argument('-o', '--output', type=str)
    args = parser.parse_args()

    cache = SimplifiedStageCache(args.cache_dir)

    if args.stats:
        print(json.dumps(cache.get_statistics(), ensure_ascii=False, indent=2))
    elif args.list:
        print('\n'.join(sorted(cache.list_all_scenes())))
    elif args.scene:
        print(json.dumps(cache.get_scene_progress(args.scene), ensure_ascii=False, indent=2))
    elif args.view:
        cache.view_cache_file(args.view)
    elif args.validate:
        print(json.dumps(cache.validate_cache(args.validate), ensure_ascii=False, indent=2))
    elif args.clear:
        cache.clear_scene_cache(args.clear)
    elif args.clear_all:
        cache.clear_all_caches()
    elif args.compact:
        cache.compact_cache(args.compact)
    elif args.export:
        cache.export_all_qa_pairs(args.export)
    elif args.export_scene:
        items = cache.export_qa_pairs(args.export_scene)
        if args.output:
            out = Path(args.output)
            out.parent.mkdir(parents=True, exist_ok=True)
            with open(out, 'w', encoding='utf-8') as file:
                json.dump(items, file, ensure_ascii=False, indent=2)
        else:
            print(json.dumps(items, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
