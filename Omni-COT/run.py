import argparse
import yaml
import json
from typing import List, Dict, Tuple
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from tqdm import tqdm
from collections import defaultdict

from src.osr_data_loader import OSRDataLoader
from src.scene_understanding import SceneUnderstanding
from src.question_generator import QuestionGenerator
from src.question_scorer import QuestionScorer
from src.cot_generator import CoTGenerator
from src.quality_judge import QualityJudge
from src.scene_data_extractor import SceneDataExtractor
from src.stage_cache import SimplifiedStageCache

QUESTION_TYPES = (
    'viewpoint_transform_identify',
    'viewpoint_transform_angle',
    'multi_hop_object',
    'multi_hop_direction',
    'move_translation',
    'move_turn_combined',
)


class SimplifiedBatchPipeline:
    
    
    def __init__(self, config: dict, batch_config: dict):
        self.config = config
        self.batch_config = batch_config
        
              
        self.stats = {
            'scenes_processed': 0,
            'scenes_failed': 0,
            'qa_generated': 0,
            'qa_passed': 0,
            'total_tokens': 0,
            'start_time': None,
            'end_time': None
        }
    
    def run(self, data_root: str, output_path: str,
            max_scenes: int = None,
            question_batch_count: int = 2,
            target_qa_per_type: int = 1,
            max_workers: int = None):
        
        self.stats['start_time'] = datetime.now()
        
        print("\n" + "="*80)
        print(" SIMPLIFIED BATCH PIPELINE v3.0")
        print("="*80)
        print(f" Data Root: {data_root}")
        print(f" Output: {output_path}")
        print(f" Question Generation: {question_batch_count} batches x {len(QUESTION_TYPES)} types = {question_batch_count * len(QUESTION_TYPES)} questions")
        print(f" Target QA Pairs: {target_qa_per_type} per type x {len(QUESTION_TYPES)} types = {target_qa_per_type * len(QUESTION_TYPES)} pairs")
        print(f" Max Workers: {max_workers or cpu_count()} processes")
        print(f" Concurrency: Scene-level ONLY (no intra-scene parallelism)")
        print("="*80 + "\n")
        
                
        data_loader = OSRDataLoader(data_root)
        scenes = data_loader.scenes
        if max_scenes:
            scenes = scenes[:max_scenes]
        
        print(f" Total scenes to process: {len(scenes)}\n")
        if len(scenes) == 0:
            print(" [WARN] No scenes discovered. Writing empty output and exiting.")
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump([], f, ensure_ascii=False, indent=2)
            self.stats['end_time'] = datetime.now()
            return
        
                  
        if max_workers is None:
            max_workers = min(cpu_count(), 5)                   
            print(f" [Auto-set] max_workers={max_workers} (API safety limit)\n")
        elif max_workers < 1:
            raise ValueError("max_workers must be >= 1")
        
                   
        if max_workers > 10:
            print(f" [Warning] max_workers={max_workers} may exceed API rate limits!")
            confirm = input("Continue anyway? (yes/no): ")
            if confirm.lower() != 'yes':
                print("Aborted.")
                return
        
        actual_workers = min(max_workers, len(scenes))
        
        all_qa_pairs = []
        
        with ProcessPoolExecutor(max_workers=actual_workers) as executor:
                        
            futures = {
                executor.submit(
                    process_single_scene_wrapper,
                    scene_dir,
                    self.config,
                    self.batch_config,
                    question_batch_count,
                    target_qa_per_type
                ): scene_dir
                for scene_dir in scenes
            }
            
                  
            with tqdm(total=len(scenes), desc="Processing scenes", unit="scene") as pbar:
                for future in as_completed(futures):
                    scene_dir = futures[future]
                    try:
                        result = future.result()
                        
                        if result['status'] == 'success':
                            all_qa_pairs.extend(result['qa_pairs'])
                            self.stats['scenes_processed'] += 1
                            self.stats['qa_generated'] += len(result['qa_pairs'])
                            self.stats['qa_passed'] += result['passed_count']
                            self.stats['total_tokens'] += result['total_tokens']
                            
                            pbar.set_postfix_str(
                                f"[OK] {scene_dir.name}: {len(result['qa_pairs'])} QAs"
                            )
                        else:
                            self.stats['scenes_failed'] += 1
                            pbar.set_postfix_str(f"[FAIL] {scene_dir.name}: {result.get('error', 'Unknown')}")
                        
                    except Exception as e:
                        self.stats['scenes_failed'] += 1
                        pbar.set_postfix_str(f"[ERROR] {scene_dir.name}: {str(e)}")
                    
                    pbar.update(1)
        
              
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_qa_pairs, f, ensure_ascii=False, indent=2)
        
              
        self.stats['end_time'] = datetime.now()
        total_time = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        print("\n" + "="*80)
        print(" PIPELINE COMPLETED")
        print("="*80)
        print(f"\n Scene Statistics:")
        print(f"  - Total Scenes: {len(scenes)}")
        print(f"  - Processed: {self.stats['scenes_processed']}")
        print(f"  - Failed: {self.stats['scenes_failed']}")
        
        print(f"\n QA Statistics:")
        print(f"  - Total Generated: {self.stats['qa_generated']}")
        print(f"  - Passed: {self.stats['qa_passed']}")
        if self.stats['qa_generated'] > 0:
            print(f"  - Pass Rate: {self.stats['qa_passed']/self.stats['qa_generated']*100:.1f}%")
        
        print(f"\n Token Usage:")
        print(f"  - Total Tokens: {self.stats['total_tokens']:,}")
        
        print(f"\n Performance:")
        print(f"  - Total Time: {total_time:.1f}s ({total_time/60:.1f} min)")
        if total_time > 0:
            print(f"  - QA Pairs/min: {self.stats['qa_generated']/(total_time/60):.2f}")
        
        print(f"\n Output: {output_path}")
        print("="*80 + "\n")


def process_single_scene_wrapper(scene_dir: Path, config: dict, batch_config: dict,
                                 question_batch_count: int, target_qa_per_type: int) -> dict:
    
    processor = SceneProcessor(config, batch_config)
    return processor.process_scene(scene_dir, question_batch_count, target_qa_per_type)


class SceneProcessor:
    
    
    def __init__(self, config: dict, batch_config: dict):
        self.config = config
        self.batch_config = batch_config
        
               
        random_seed = batch_config.get('batch', {}).get('random_seed', 42)
        self.scene_data_extractor = SceneDataExtractor(random_seed=random_seed)
        self.scene_understander = SceneUnderstanding(config)
        self.question_gen = QuestionGenerator(config)
        self.question_scorer = QuestionScorer(config)
        self.cot_generator = CoTGenerator(config)
        self.quality_judge = QualityJudge(config)
        
               
        cache_dir = (
            batch_config.get('batch', {})
            .get('cache', {})
            .get('stage_cache_dir', 'data/cache/stage_cache')
        )
        self.cache = SimplifiedStageCache(cache_dir)
    
    def process_scene(self, scene_dir: Path, question_batch_count: int, target_qa_per_type: int) -> dict:
        
        scene_id = scene_dir.name
        
        try:
            print(f"\n[{scene_id}] Starting processing...")
            
                                               
            description, random_objects = self._understand_scene(scene_dir, scene_id)
            if not description:
                return {'status': 'failed', 'scene_id': scene_id, 'error': 'Scene understanding failed'}
            
                                            
            questions_by_type = self._generate_and_select_questions(
                description, scene_id, question_batch_count, target_qa_per_type
            )
            
            if not questions_by_type:
                return {'status': 'failed', 'scene_id': scene_id, 'error': 'No valid questions'}
            
                                          
            qa_pairs = self._generate_qa_pairs_with_adaptive_questions(
                description, scene_id, questions_by_type, question_batch_count, target_qa_per_type
            )
            
                
            passed_count = sum(
                1 for qa in qa_pairs 
                if qa.get('quality_evaluation', {}).get('passed', False)
            )
            total_tokens = sum(
                qa.get('token_usage', {}).get('total', 0) 
                for qa in qa_pairs
            )
            
            print(f"[{scene_id}] [OK] Completed: {len(qa_pairs)} QA pairs ({passed_count} passed)")
            
            return {
                'status': 'success',
                'scene_id': scene_id,
                'qa_pairs': qa_pairs,
                'passed_count': passed_count,
                'total_tokens': total_tokens,
                'random_objects': random_objects               
            }
            
        except Exception as e:
            print(f"[{scene_id}] [FAIL] Failed: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'status': 'failed',
                'scene_id': scene_id,
                'error': str(e)
            }
    
    def _understand_scene(self, scene_dir: Path, scene_id: str) -> Tuple[str, List[Dict]]:
        
              
        cached = self.cache.get_stage_1(scene_id)
        if cached:
            print(f"[{scene_id}] Stage 1: [CACHE] Loaded from cache")
            return cached['description'], cached.get('random_objects', [])
        
                           
        scene_data_path = scene_dir / 'scene_data.json'
        if not scene_data_path.exists():
            raise FileNotFoundError(f"scene_data.json not found in {scene_dir}")
        
                        
        extracted_data = self.scene_data_extractor.extract(str(scene_data_path))
        scene_data_text = self.scene_data_extractor.format_for_prompt(extracted_data)
        random_objects = extracted_data.get('random_objects', [])
        
                   
        description, usage = self.scene_understander.understand_scene_from_json(scene_data_text)
        
                      
        self.cache.save_stage_1(scene_id, description, usage, random_objects)
        
        print(f"[{scene_id}] Stage 1: [OK] Scene understood ({usage['total_tokens']} tokens)")
        print(f"[{scene_id}] Stage 1: [INFO] Selected {len(random_objects)} random objects")
        
        return description, random_objects
    
    def _generate_and_select_questions(self, description: str, scene_id: str,
                                      question_batch_count: int, target_qa_per_type: int) -> dict:
               
        cached = self.cache.get_stage_3(scene_id)
        if cached and self._validate_cached_questions(cached, target_qa_per_type):
            print(f"[{scene_id}] Stage 2-3: [CACHE] Loaded from cache")
            return cached
        
               
        questions_by_type = {qtype: [] for qtype in QUESTION_TYPES}
        
                  
        print(f"[{scene_id}] Stage 2: Generating {question_batch_count} batches x {len(QUESTION_TYPES)} types...")
        all_questions, _ = self.question_gen.generate_questions_batch(
            description, batch_count=question_batch_count
        )
        
            
        print(f"[{scene_id}] Stage 3: Scoring {len(all_questions)} questions...")
        scored_questions, _ = self.question_scorer.score_questions(
            description, all_questions
        )
        
               
        for q in scored_questions:
            if q['score'] >= 5.0 and q.get('type') in questions_by_type:
                questions_by_type[q['type']].append(q)
        
                 
        for qtype in QUESTION_TYPES:
            questions_by_type[qtype].sort(key=lambda entry: entry['score'], reverse=True)
        
                 
        for qtype in QUESTION_TYPES:
            attempt = 0
            max_attempts = 3          
            
            while len(questions_by_type[qtype]) < target_qa_per_type and attempt < max_attempts:
                print(f"[{scene_id}] Stage 2: {qtype} insufficient ({len(questions_by_type[qtype])}/{target_qa_per_type}), generating more...")
                
                             
                new_questions = self._generate_questions_of_type(description, qtype, question_batch_count)
                
                if not new_questions:
                    print(f"[{scene_id}] [WARN] Failed to generate more questions for {qtype}")
                    break
                
                    
                scored_new, _ = self.question_scorer.score_questions(description, new_questions)
                
                        
                valid_new = [q for q in scored_new if q['score'] >= 5.0]
                questions_by_type[qtype].extend(valid_new)
                questions_by_type[qtype].sort(key=lambda entry: entry['score'], reverse=True)
                
                attempt += 1
        
                
        for qtype in QUESTION_TYPES:
            questions_by_type[qtype] = questions_by_type[qtype][:target_qa_per_type]
            print(f"[{scene_id}]   {qtype}: {len(questions_by_type[qtype])}/{target_qa_per_type} selected")
        
              
        self.cache.save_stage_3(scene_id, questions_by_type)
        
        return questions_by_type
    
    def _generate_questions_of_type(self, description: str, qtype: str, count: int) -> list:
        
        try:
                                                     
            if hasattr(self.question_gen, 'generate_specific_type_question'):
                questions, _ = self.question_gen.generate_specific_type_question(
                    scene_description=description,                            
                    question_type=qtype,                                  
                    count=count
                )
                return questions
            else:
                                    
                                                      
                print(f"[WARN] generate_specific_type_question not found, using fallback method")
                all_questions, _ = self.question_gen.generate_questions_batch(
                    scene_description=description,
                    batch_count=1               
                )
                
                            
                typed_questions = [
                    q for q in all_questions
                    if q.get('type') == qtype
                ]
                
                           
                while len(typed_questions) < count:
                    additional_questions, _ = self.question_gen.generate_questions_batch(
                        scene_description=description,
                        batch_count=1
                    )
                    typed_additional = [
                        q for q in additional_questions
                        if q.get('type') == qtype
                    ]
                    typed_questions.extend(typed_additional)
                
                return typed_questions[:count]
        
        except Exception as e:
            print(f"Error generating questions of type {qtype}: {e}")
            return []
    
    def _validate_cached_questions(self, cached_questions: dict, target_qa_per_type: int) -> bool:
        for qtype in QUESTION_TYPES:
            questions = cached_questions.get(qtype, [])
            if len(questions) < target_qa_per_type:
                return False
        return True
    
    def _generate_qa_pairs_with_adaptive_questions(
        self, description: str, scene_id: str,
        questions_by_type: dict, question_batch_count: int, target_qa_per_type: int
    ) -> list:
        target_count = target_qa_per_type * len(QUESTION_TYPES)
        qa_pairs = []
        used_questions = set()            
        
        print(f"[{scene_id}] Stage 4-5: Generating QA pairs (target: {target_count})...")
        
        max_rounds = 20
        round_num = 0
        consecutive_no_progress = 0
        
        while len(qa_pairs) < target_count and round_num < max_rounds:
            round_num += 1
            
                         
            type_counts = defaultdict(int)
            for qa in qa_pairs:
                qtype = qa['question_data'].get('type', 'unknown')
                type_counts[qtype] += 1
            
                       
            deficit_types = [
                qtype for qtype in QUESTION_TYPES
                if type_counts[qtype] < target_qa_per_type
            ]
            
            if not deficit_types:
                break            
            
            print(f"[{scene_id}] Round {round_num}: "
                  f"{len(qa_pairs)}/{target_count} QAs, "
                  f"{len(deficit_types)} types need more")
            
            made_progress = False
            
                            
            for qtype in deficit_types:
                                  
                available_questions = [
                    q for q in questions_by_type.get(qtype, [])
                    if q['question'] not in used_questions
                ]
                
                                 
                if len(available_questions) == 0:
                    needed_qa = target_qa_per_type - type_counts[qtype]           
                    
                    print(f"[{scene_id}]   [WARN] {qtype}: No available questions, "
                          f"need {needed_qa} more QAs")
                    print(f"[{scene_id}]   [GEN] Generating {question_batch_count} new questions for {qtype}...")
                    
                                       
                    new_raw_questions = self._generate_questions_of_type(
                        description, qtype, count=question_batch_count
                    )
                    
                    if not new_raw_questions:
                        print(f"[{scene_id}]   [FAIL] Failed to generate questions")
                        continue
                    
                    print(f"[{scene_id}]   [OK] Generated {len(new_raw_questions)} questions")
                    
                                
                    scored_new_questions, _ = self.question_scorer.score_questions(
                        description, new_raw_questions
                    )
                    
                                       
                    valid_questions = [
                        q for q in scored_new_questions
                        if q['score'] >= 5.0
                    ]
                    
                    if not valid_questions:
                        print(f"[{scene_id}]   [FAIL] No questions passed score threshold")
                        continue
                    
                                         
                    valid_questions.sort(key=lambda q: q['score'], reverse=True)
                    top_selected_questions = valid_questions[:target_qa_per_type]
                    
                    scores_str = ', '.join([f"{q['score']:.1f}" for q in top_selected_questions])
                    print(f"[{scene_id}]   [OK] Selected top {len(top_selected_questions)}/{target_qa_per_type} questions "
                          f"(scores: [{scores_str}])")
                    
                                     
                    if qtype not in questions_by_type:
                        questions_by_type[qtype] = []
                    questions_by_type[qtype].extend(top_selected_questions)
                    
                              
                    available_questions = top_selected_questions
                
                             
                for question in available_questions:
                    if type_counts[qtype] >= target_qa_per_type:
                        break          
                    
                    question_text = question['question']
                    if question_text in used_questions:
                        continue
                    
                                      
                    qa = self._generate_single_qa(description, scene_id, question)
                    
                    if qa and qa['quality_evaluation']['passed']:
                        qa_pairs.append(qa)
                        used_questions.add(question_text)
                        type_counts[qtype] += 1
                        made_progress = True
                        consecutive_no_progress = 0
                        
                        score = qa['quality_evaluation']['overall_score']
                        print(f"[{scene_id}]   [OK] QA #{len(qa_pairs)}/{target_count} "
                              f"({qtype}, score={score:.1f})")
                    else:
                        print(f"[{scene_id}]   [FAIL] QA failed quality check ({qtype})")
            
                     
            if not made_progress:
                consecutive_no_progress += 1
                print(f"[{scene_id}] [WARN] No progress in round {round_num} "
                      f"(consecutive: {consecutive_no_progress})")
                
                if consecutive_no_progress >= 3:
                    print(f"[{scene_id}] [STOP] No progress for 3 rounds, stopping early")
                    break
        
              
        self._print_final_qa_summary(scene_id, qa_pairs, target_count, QUESTION_TYPES, target_qa_per_type)
        
        return qa_pairs
    
    def _print_final_qa_summary(self, scene_id: str, qa_pairs: list,
                               target_count: int, question_types: list, target_qa_per_type: int):
        type_counts = defaultdict(int)
        for qa in qa_pairs:
            type_counts[qa['question_data']['type']] += 1
        
        print(f"\n{'='*70}")
        if len(qa_pairs) < target_count:
            print(f"[{scene_id}] [WARN] Generated {len(qa_pairs)}/{target_count} QA pairs")
        else:
            print(f"[{scene_id}] [OK] Generated {len(qa_pairs)}/{target_count} QA pairs")
        
        print(f"[{scene_id}] Type distribution:")
        for qtype in question_types:
            count = type_counts[qtype]
            if count >= target_qa_per_type:
                status = "[OK]"
            elif count > 0:
                status = f"[PARTIAL] ({count}/{target_qa_per_type})"
            else:
                status = f"[FAIL] (0/{target_qa_per_type})"
            print(f"[{scene_id}]   {status} {qtype}")
        print(f"{'='*70}\n")
    
    def _generate_single_qa(self, description: str, scene_id: str,
                           question: dict) -> dict:
        
        question_text = question['question']
        question_type = question.get('type', 'unknown')
        
              
        cached_qa = self.cache.get_qa_pair(scene_id, question_text)
        if cached_qa and cached_qa['stage_5']['status'] == 'completed':
            return self._convert_cached_qa(cached_qa, question)
        
        try:
                            
            initial_reasoning, reasoning_usage = self.cot_generator.generate_reasoning(
                description, question_text, question
            )
            
            summarized_answer, summarize_usage = self.cot_generator.summarize_answer(
                initial_reasoning, question_text, question_type
            )
            
                  
            parsed = self._parse_summarized_answer(summarized_answer)
            
                           
            evaluation = self.quality_judge.evaluate(
                description,
                question_text,
                initial_reasoning,
                parsed['final_answer'],
                parsed['cot'],
                question_metadata=question
            )
            
                     
            total_tokens = (
                reasoning_usage['total_tokens'] +
                summarize_usage['total_tokens'] +
                evaluation.get('usage', {}).get('total_tokens', 0)
            )
            
                  
            qa_data = {
                'scene_id': scene_id,
                'question_data': question,
                'reasoning_process': {
                    'initial_reasoning': initial_reasoning,
                    'structured_cot': initial_reasoning
                },
                'answer_data': {
                    'final_answer': parsed['final_answer'],
                    'cot': parsed['cot'],
                },
                'quality_evaluation': {
                    'reasoning_score': evaluation.get('reasoning_score', 0),
                    'answer_score': evaluation.get('answer_score', 0),
                    'overall_score': evaluation.get('overall_score', 0),
                    'passed': evaluation.get('passed', False)
                },
                'token_usage': {
                    'total': total_tokens
                },
                'timestamp': datetime.now().isoformat()
            }
            
                  
            self.cache.save_qa_pair(scene_id, question_text, qa_data)
            
            return qa_data
            
        except Exception as e:
            print(f"[{scene_id}] Error generating QA: {str(e)}")
            return None
    
    def _parse_summarized_answer(self, summarized_text: str) -> dict:
        
        import re
        
        result = {
            'final_answer': '',
            'cot': []
        }
        
                                              
        
                                                  
        answer_patterns = [
            r'(?:Final\s+)?Answer:\s*(.+?)(?=\n\n|COT:|Chain of Thought:|$)',                 
            r'(?:Final\s+)?Answer:\s*(.+)',           
        ]
        
        for pattern in answer_patterns:
            answer_match = re.search(pattern, summarized_text, re.IGNORECASE | re.DOTALL)
            if answer_match:
                answer_text = answer_match.group(1).strip()
                                   
                answer_text = re.sub(r'\s+', ' ', answer_text)
                result['final_answer'] = answer_text
                break
        
                                               
        
                                           
        cot_patterns = [
            r'COT:\s*(.+)$',
        ]
        
        cot_text = None
        for pattern in cot_patterns:
            cot_match = re.search(pattern, summarized_text, re.IGNORECASE | re.DOTALL)
            if cot_match:
                cot_text = cot_match.group(1)
                break
        
        if cot_text:
            steps = []
            for line in cot_text.split('\n'):
                line = line.strip()
                if not line:
                    continue
                          
                line = re.sub(r'^[\d]+\.\s*', '', line)           
                line = re.sub(r'^[-*•]\s*', '', line)                            
                line = re.sub(r'^\([a-z]\)\s*', '', line)          
                if line:
                    steps.append(line)
            result['cot'] = steps
        
                                               
        
        if not result['final_answer']:
                                         
            first_line = summarized_text.split('\n')[0].strip()
            if first_line and len(first_line) < 200:           
                result['final_answer'] = first_line
            else:
                result['final_answer'] = "Failed to extract answer"
        
        if not result['cot']:
            result['cot'] = ["No COT extracted"]
        
        return result



    def _convert_cached_qa(self, cached_qa: dict, question: dict) -> dict:
        stage4 = cached_qa['stage_4']
        stage5 = cached_qa['stage_5']
        
        return {
            'scene_id': cached_qa['scene_id'],
            'question_data': question,
            'reasoning_process': stage4['reasoning_process'],
            'answer_data': stage4['answer_data'],
            'quality_evaluation': stage5['evaluation'],
            'token_usage': {
                'total': stage4['tokens'] + stage5['tokens']
            },
            'timestamp': stage5['timestamp']
        }


def main():
    parser = argparse.ArgumentParser(
        description='Simplified Batch Pipeline v3.0 - Scene-level Process Parallelism',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--data-root', type=str, required=True,
                       help='Root directory containing OSR scenes')
    parser.add_argument('--output', type=str,
                       default='data/outputs/simplified_batch_output.json',
                       help='Output JSON file path')
    parser.add_argument('--config', type=str,
                       default='config/api_config.yaml',
                       help='API configuration file')
    parser.add_argument('--batch-config', type=str,
                       default='config/batch_config.yaml',
                       help='Batch configuration file')
    parser.add_argument('--max-scenes', type=int, default=None,
                       help='Maximum number of scenes to process')
    parser.add_argument('--question-batches', type=int, default=2,
                       help='Initial question generation batches (default: 2)')
    parser.add_argument('--target-qa-per-type', type=int, default=1,
                       help='Target accepted QA pairs per type (default: 1)')
    parser.add_argument('-x', dest='question_batches', type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument('-y', '--questions-per-type', dest='target_qa_per_type', type=int, default=argparse.SUPPRESS, help=argparse.SUPPRESS)
    parser.add_argument('--max-workers', type=int, default=None,
                       help='Maximum parallel processes (default: min(CPU count, 5))')
    
    args = parser.parse_args()
    
          
    if not Path(args.data_root).exists():
        print(f"[ERROR] Data root directory not found: {args.data_root}")
        return
    
    if args.question_batches < 1:
        print(f"[ERROR] Invalid question-batches: {args.question_batches} (must be >= 1)")
        return
    
    if args.target_qa_per_type < 1:
        print(f"[ERROR] Invalid target-qa-per-type: {args.target_qa_per_type} (must be >= 1)")
        return

    if args.max_workers is not None and args.max_workers < 1:
        print(f"[ERROR] Invalid max-workers: {args.max_workers} (must be >= 1)")
        return
    
          
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"[OK] Loaded API config from {args.config}")
    except Exception as e:
        print(f"[ERROR] Failed to load API config: {e}")
        return
    
    try:
        with open(args.batch_config, 'r', encoding='utf-8') as f:
            batch_config = yaml.safe_load(f)
        print(f"[OK] Loaded batch config from {args.batch_config}")
    except Exception as e:
        print(f"[ERROR] Failed to load batch config: {e}")
        return
    
                
    pipeline = SimplifiedBatchPipeline(config, batch_config)
    
    try:
        pipeline.run(
            data_root=args.data_root,
            output_path=args.output,
            max_scenes=args.max_scenes,
            question_batch_count=args.question_batches,
            target_qa_per_type=args.target_qa_per_type,
            max_workers=args.max_workers
        )
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print(" PIPELINE INTERRUPTED BY USER")
        print("="*80)
        print(" Progress has been saved to cache.")
        print(" You can resume by running the same command again.")
        print("="*80 + "\n")
    except Exception as e:
        print("\n\n" + "="*80)
        print(" PIPELINE EXECUTION FAILED")
        print("="*80)
        print(f"Error: {str(e)}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        print("="*80 + "\n")


if __name__ == "__main__":
    main()
