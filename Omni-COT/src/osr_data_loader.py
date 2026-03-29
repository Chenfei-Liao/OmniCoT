                        
import json
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

class OSRDataLoader:
    
    def __init__(self, data_root: str):
        self.data_root = Path(data_root)
        self.scenes = self._discover_scenes()
    
    def _discover_scenes(self) -> List[Path]:
        
        scenes = []
        for scene_dir in self.data_root.iterdir():
            if not scene_dir.is_dir():
                continue
            
                                       
            has_json = (scene_dir / 'scene_data.json').exists()
            has_rgb = (scene_dir / 'rgb.png').exists()
            
            if has_json or has_rgb:
                scenes.append(scene_dir)
        
        print(f"Discovered {len(scenes)} scenes in {self.data_root}")
        return sorted(scenes)
    
    def _safe_pickle_load(self, pkl_file: Path) -> Dict:
        
        try:
            with open(pkl_file, 'rb') as f:
                return pickle.load(f)
        except (ModuleNotFoundError, AttributeError):
            return None
        except Exception:
            return None
    
    def load_scene(self, scene_dir: Path) -> Dict:
        
        
        scene_data = {
            'scene_id': scene_dir.name,
            'scene_dir': str(scene_dir),
            'rgb_path': str(scene_dir / 'rgb.png'),
            'depth_path': str(scene_dir / 'depth.png'),
            'layout_path': str(scene_dir / 'room_layout.jpg'),
        }
        
                                                        
        cogmap_file = scene_dir / 'cognitive_map.json'
        if cogmap_file.exists():
            with open(cogmap_file, 'r', encoding='utf-8') as f:
                scene_data['cognitive_map'] = json.load(f)
        
                                             
        pkl_file = scene_dir / 'data.pkl'
        if pkl_file.exists():
            layout_data = self._safe_pickle_load(pkl_file)
            if layout_data is not None:
                scene_data['layout_data'] = layout_data
        
                                          
        qa_file = scene_dir / f"{scene_dir.name}_qa.json"
        if qa_file.exists():
            try:
                with open(qa_file, 'r', encoding='utf-8') as f:
                    scene_data['reference_qa'] = json.load(f)
            except Exception:
                scene_data['reference_qa'] = None
        
        return scene_data
    
    def extract_objects(self, scene_data: Dict) -> List[Dict]:
        
        objects = []
        
        if 'cognitive_map' not in scene_data:
            return self._generate_default_objects(scene_data)
        
        cogmap_data = scene_data['cognitive_map']
        
                                                          
        if 'cogmap_coordinate' in cogmap_data:
            for obj_class, coordinates in cogmap_data['cogmap_coordinate'].items():
                for idx, coord in enumerate(coordinates):
                    row, col = coord
                    obj_id = f"{obj_class}_{idx}" if len(coordinates) > 1 else obj_class
                    
                    objects.append({
                        'id': obj_id,
                        'name': obj_class,
                        'category': self._infer_category(obj_class),
                        'grid_position': {'row': row, 'col': col},
                        'position': coord,                       
                        'relations': []
                    })
        
                                            
        elif 'cognitive_map' in cogmap_data and isinstance(cogmap_data['cognitive_map'], list):
            grid = cogmap_data['cognitive_map']
            for row_idx, row in enumerate(grid):
                for col_idx, cell in enumerate(row):
                    if cell:                         
                        for obj_name in cell:
                            obj_id = f"{obj_name}_{row_idx}_{col_idx}"
                            objects.append({
                                'id': obj_id,
                                'name': obj_name,
                                'category': self._infer_category(obj_name),
                                'grid_position': {'row': row_idx, 'col': col_idx},
                                'position': [row_idx, col_idx],
                                'relations': []
                            })
        
                                                           
        elif 'class_count' in cogmap_data:
            for obj_class, count in cogmap_data['class_count'].items():
                for idx in range(count):
                    obj_id = f"{obj_class}_{idx}" if count > 1 else obj_class
                    objects.append({
                        'id': obj_id,
                        'name': obj_class,
                        'category': self._infer_category(obj_class),
                        'relations': []
                    })
        
        if not objects:
            return self._generate_default_objects(scene_data)
        
        return objects
    
    def _infer_category(self, obj_name: str) -> str:
        
        furniture = ['chair', 'table', 'sofa', 'bed', 'desk', 'cabinet', 'shelf', 
                     'bookshelf', 'stool', 'bench', 'couch', 'dresser', 'sofa_chair']
        appliance = ['tv', 'lamp', 'monitor', 'computer', 'refrigerator', 'oven', 
                     'microwave', 'dishwasher', 'piano']
        structure = ['door', 'window', 'wall', 'ceiling', 'floor']
        decoration = ['picture', 'painting', 'plant', 'vase', 'sculpture', 'mirror']
        textile = ['carpet', 'rug', 'curtain', 'cushion', 'pillow']
        
        obj_lower = obj_name.lower()
        
        if any(f in obj_lower for f in furniture):
            return 'furniture'
        elif any(a in obj_lower for a in appliance):
            return 'appliance'
        elif any(s in obj_lower for s in structure):
            return 'structure'
        elif any(d in obj_lower for d in decoration):
            return 'decoration'
        elif any(t in obj_lower for t in textile):
            return 'textile'
        else:
            return 'object'
    
    def extract_spatial_relations(self, scene_data: Dict) -> List[Dict]:
        
        objects = self.extract_objects(scene_data)
        relations = []
        
                                                 
        for i, obj1 in enumerate(objects):
            if 'grid_position' not in obj1:
                continue
            
            pos1 = obj1['grid_position']
            
            for j, obj2 in enumerate(objects):
                if i >= j or 'grid_position' not in obj2:
                    continue
                
                pos2 = obj2['grid_position']
                
                                                
                relation = self._calculate_spatial_relation(pos1, pos2)
                
                if relation:
                    relations.append({
                        'source': obj1['name'],
                        'target': obj2['name'],
                        'relation': relation,
                        'source_id': obj1['id'],
                        'target_id': obj2['id'],
                        'distance': self._manhattan_distance(pos1, pos2)
                    })
        
        return relations
    
    def _calculate_spatial_relation(self, pos1: Dict, pos2: Dict) -> str:
        
        row_diff = pos2['row'] - pos1['row']
        col_diff = pos2['col'] - pos1['col']
        
                                           
        if abs(row_diff) + abs(col_diff) == 1:
            if row_diff == -1:
                return 'above'
            elif row_diff == 1:
                return 'below'
            elif col_diff == -1:
                return 'left of'
            elif col_diff == 1:
                return 'right of'
        
                                       
        elif abs(row_diff) + abs(col_diff) == 2:
            return 'near'
        
                  
        elif abs(row_diff) == 1 and abs(col_diff) == 1:
            if row_diff < 0 and col_diff < 0:
                return 'upper-left of'
            elif row_diff < 0 and col_diff > 0:
                return 'upper-right of'
            elif row_diff > 0 and col_diff < 0:
                return 'lower-left of'
            else:
                return 'lower-right of'
        
                                           
        elif abs(row_diff) > abs(col_diff):
            if row_diff < 0:
                return 'in front of'
            else:
                return 'behind'
        elif abs(col_diff) > abs(row_diff):
            if col_diff < 0:
                return 'to the left of'
            else:
                return 'to the right of'
        
        return None
    
    def _manhattan_distance(self, pos1: Dict, pos2: Dict) -> int:
        
        return abs(pos1['row'] - pos2['row']) + abs(pos1['col'] - pos2['col'])
    
    def _generate_default_objects(self, scene_data: Dict) -> List[Dict]:
        
        print(f"   No objects found for scene {scene_data['scene_id']}, generating defaults")
        
        return [
            {'id': 'floor', 'name': 'floor', 'category': 'structure'},
            {'id': 'ceiling', 'name': 'ceiling', 'category': 'structure'},
            {'id': 'walls', 'name': 'walls', 'category': 'structure'},
            {'id': 'room', 'name': 'room', 'category': 'space'},
        ]
    
    def get_scene_summary(self, scene_data: Dict) -> str:
        
        objects = self.extract_objects(scene_data)
        relations = self.extract_spatial_relations(scene_data)
        
                                          
        grid_info = ""
        if 'cognitive_map' in scene_data and 'cognitive_map' in scene_data['cognitive_map']:
            grid = scene_data['cognitive_map']['cognitive_map']
            if isinstance(grid, list):
                grid_info = f"Grid size: {len(grid)}x{len(grid[0])}\n"
        
        summary = f"Scene ID: {scene_data['scene_id']}\n"
        summary += grid_info
        summary += f"Total objects: {len(objects)}\n"
        
        if objects:
            obj_names = [obj.get('name', 'unknown') for obj in objects[:15]]
            summary += f"Objects: {', '.join(obj_names)}\n"
        
        if relations:
            summary += f"\nSpatial relations: {len(relations)} connections\n"
                                               
            for rel in relations[:5]:
                summary += f"  - {rel['source']} {rel['relation']} {rel['target']}\n"
        
        return summary
