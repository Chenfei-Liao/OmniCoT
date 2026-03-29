
from collections import defaultdict
from datetime import datetime
from typing import Dict, List
import json
from pathlib import Path

class ErrorTracker:
    
    
    def __init__(self):
        self.errors = []
        self.error_stats = defaultdict(int)
        self.retry_stats = defaultdict(int)
    
    def log_error(self, component: str, error_type: str, error_msg: str, 
                  item_id: str = None, retried: bool = False):
        
        error_entry = {
            'timestamp': datetime.now().isoformat(),
            'component': component,
            'error_type': error_type,
            'error_message': error_msg[:500],
            'item_id': item_id,
            'retried': retried
        }
        
        self.errors.append(error_entry)
        self.error_stats[f"{component}_{error_type}"] += 1
        
        if retried:
            self.retry_stats[component] += 1
    
    def get_statistics(self) -> Dict:
        
        return {
            'total_errors': len(self.errors),
            'error_breakdown': dict(self.error_stats),
            'retry_breakdown': dict(self.retry_stats),
            'most_common_errors': sorted(
                self.error_stats.items(), 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
        }
    
    def save_report(self, output_path: str):
        
        report = {
            'summary': self.get_statistics(),
            'detailed_errors': self.errors
        }
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n Error report saved to: {output_path}")