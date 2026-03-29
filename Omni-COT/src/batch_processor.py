import asyncio
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Tuple
from pathlib import Path
import json
from datetime import datetime

class RateLimiter:
    
    def __init__(self, requests_per_second: int = 20):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time = 0
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        
        async with self.lock:
            current_time = time.time()
            time_since_last = current_time - self.last_request_time
            
            if time_since_last < self.min_interval:
                await asyncio.sleep(self.min_interval - time_since_last)
            
            self.last_request_time = time.time()

class BatchProcessor:
    
    
    def __init__(self, config: dict):
        self.config = config
        self.rate_limiter = RateLimiter(
            config['batch']['rate_limit']['requests_per_second']
        )
        self.max_workers = config['batch']['qa_generation_workers']
    
    def process_batch_parallel(self, items: List, process_func: Callable, 
                              max_workers: int = None) -> Tuple[List[Any], List[Dict[str, str]]]:
        
        if max_workers is None:
            max_workers = self.max_workers
        
        results = []
        errors = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
              
            future_to_item = {
                executor.submit(process_func, item): item 
                for item in items
            }
            
              
            for future in as_completed(future_to_item):
                item = future_to_item[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    error_info = {
                        'item': str(item),
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                    errors.append(error_info)
                    print(f"  [Error] Processing failed: {error_info}")
        
        return results, errors
    
    def save_checkpoint(self, data: List[Dict], checkpoint_path: Path):
        
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        with open(checkpoint_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  [Checkpoint] Saved to {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: Path) -> List[Dict]:
        
        if not checkpoint_path.exists():
            return []
        
        with open(checkpoint_path, 'r', encoding='utf-8') as f:
            return json.load(f)
