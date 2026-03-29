import hashlib
import json
import math
import random
from typing import Dict, List, Tuple


class SceneDataExtractor:
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed

    def extract(self, scene_data_path: str) -> Dict:
        with open(scene_data_path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        camera_pos_raw = data['camera']['pos']
        camera_pos = [camera_pos_raw[0], -camera_pos_raw[1], camera_pos_raw[2]]

        layout = data['layout']
        wall_vertices = self._extract_wall_vertices(layout['manhattan_world'])
        room_dims = self._calculate_room_dimensions(wall_vertices)
        objects = self._extract_objects(data.get('objs', []))
        scene_name = data.get('name', 'unknown')
        random_objects = self._select_random_objects(objects, scene_name=scene_name, count=3)

        return {
            'camera_pos': camera_pos,
            'room_dimensions': room_dims,
            'wall_vertices': wall_vertices,
            'objects': objects,
            'scene_name': scene_name,
            'random_objects': random_objects,
        }

    def _extract_wall_vertices(self, manhattan_world: List[List[float]]) -> List[Tuple[float, float, float]]:
        n = len(manhattan_world) // 2
        bottom_vertices = manhattan_world[:n]
        return [(v[0], -v[1], v[2]) for v in bottom_vertices]

    def _calculate_room_dimensions(self, vertices: List[Tuple[float, float, float]]) -> Dict:
        xs = [v[0] for v in vertices]
        ys = [v[1] for v in vertices]
        zs = [v[2] for v in vertices]
        return {
            'x_min': min(xs),
            'x_max': max(xs),
            'y_min': min(ys),
            'y_max': max(ys),
            'z_min': min(zs),
            'z_max': 2.4,
            'width': max(xs) - min(xs),
            'depth': max(ys) - min(ys),
            'height': 2.4,
        }

    def _extract_objects(self, objs: List[Dict]) -> List[Dict]:
        objects = []
        for obj in objs:
            if 'bdb3d' not in obj:
                continue
            bdb3d = obj['bdb3d']
            centroid = bdb3d['centroid']
            size = bdb3d['size']
            objects.append(
                {
                    'id': obj.get('id', -1),
                    'class_name': obj.get('classname', 'unknown'),
                    'position': {'x': centroid[0], 'y': -centroid[1], 'z': centroid[2]},
                    'size': {'width': size[0], 'depth': size[1], 'height': size[2]},
                    'basis': bdb3d.get('basis', None),
                }
            )
        return objects

    def _select_random_objects(self, objects: List[Dict], scene_name: str, count: int = 3) -> List[str]:
        if not objects:
            return []
        actual_count = min(count, len(objects))
        seed_material = f'{self.random_seed}:{scene_name}'
        scene_seed = int(hashlib.sha256(seed_material.encode('utf-8')).hexdigest()[:16], 16)
        selected = random.Random(scene_seed).sample(objects, actual_count)
        return [f"{obj['class_name']}({obj['position']['x']:.1f}, {obj['position']['y']:.1f})" for obj in selected]

    def _calculate_distance_2d(self, p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    def _calculate_distance_3d(self, p1: Dict, p2: Dict) -> float:
        return math.sqrt((p1['x'] - p2['x']) ** 2 + (p1['y'] - p2['y']) ** 2 + (p1['z'] - p2['z']) ** 2)

    def _get_direction_vector(self, v1: Tuple[float, float], v2: Tuple[float, float]) -> Tuple[float, float]:
        dx = v2[0] - v1[0]
        dy = v2[1] - v1[1]
        length = math.sqrt(dx ** 2 + dy ** 2)
        if length == 0:
            return (0.0, 0.0)
        return (dx / length, dy / length)

    def _vector_to_cardinal_direction(self, vec: Tuple[float, float]) -> str:
        x, y = vec
        degree = math.degrees(math.atan2(y, x))
        if degree < 0:
            degree += 360
        if 337.5 <= degree or degree < 22.5:
            return 'east'
        if 22.5 <= degree < 67.5:
            return 'northeast'
        if 67.5 <= degree < 112.5:
            return 'north'
        if 112.5 <= degree < 157.5:
            return 'northwest'
        if 157.5 <= degree < 202.5:
            return 'west'
        if 202.5 <= degree < 247.5:
            return 'southwest'
        if 247.5 <= degree < 292.5:
            return 'south'
        return 'southeast'

    def _find_nearest_objects(self, target_obj: Dict, all_objects: List[Dict], count: int = 3) -> List[Tuple[Dict, float]]:
        target_pos = target_obj['position']
        distances = []
        for obj in all_objects:
            if obj['id'] == target_obj['id']:
                continue
            dist = self._calculate_distance_3d(target_pos, obj['position'])
            distances.append((obj, dist))
        distances.sort(key=lambda item: item[1])
        return distances[:count]

    def generate_structured_description(self, extracted_data: Dict) -> str:
        camera_pos = extracted_data['camera_pos']
        room_dims = extracted_data['room_dimensions']
        wall_vertices = extracted_data['wall_vertices']
        objects = extracted_data['objects']

        lines = []
        lines.append('SCENE DESCRIPTION (Structured Layout)')
        lines.append('Spatial Layout Summary:')
        lines.append(f"  Wall vertices: {len(wall_vertices)}")
        lines.append(f"  Total objects: {len(objects)}")
        lines.append('')
        lines.append('Coordinate System:')
        lines.append('  Origin: (0.0, 0.0, 0.0)')
        lines.append('  X-axis: EAST, Y-axis: NORTH')
        lines.append(f"  Camera Position: ({camera_pos[0]:.1f}, {camera_pos[1]:.1f}, {camera_pos[2]:.1f})")
        lines.append(f"  Room Dimensions: {room_dims['width']:.1f} x {room_dims['depth']:.1f} x {room_dims['height']:.1f}")
        lines.append('')

        lines.append('Wall Boundary (Counter-Clockwise):')
        for index, vertex in enumerate(wall_vertices, start=1):
            lines.append(f"  V{index}: ({vertex[0]:.1f}, {vertex[1]:.1f}, {vertex[2]:.1f})")
        lines.append('')

        lines.append(f'Objects (Total: {len(objects)}):')
        for index, obj in enumerate(objects, start=1):
            pos = obj['position']
            size = obj['size']
            obj_id = f"{obj['class_name']}({pos['x']:.1f},{pos['y']:.1f})"
            lines.append(f'  {index}. {obj_id}:')
            lines.append(f"     - Size: {size['width']:.1f} x {size['depth']:.1f} x {size['height']:.1f}")
            dist_to_camera = self._calculate_distance_3d(pos, {'x': camera_pos[0], 'y': camera_pos[1], 'z': camera_pos[2]})
            lines.append(f'     - Distance to Camera: {dist_to_camera:.1f}')

            nearby = self._find_nearest_objects(obj, objects, count=3)
            if nearby:
                nearby_parts = []
                for nearby_obj, dist in nearby:
                    near_pos = nearby_obj['position']
                    direction = self._vector_to_cardinal_direction(
                        self._get_direction_vector((pos['x'], pos['y']), (near_pos['x'], near_pos['y']))
                    )
                    nearby_id = f"{nearby_obj['class_name']}({near_pos['x']:.1f},{near_pos['y']:.1f})"
                    nearby_parts.append(f'{nearby_id} ({dist:.1f}, direction: {direction})')
                lines.append(f"     - Nearby: {', '.join(nearby_parts)}")

        return '\n'.join(lines)

    def format_for_prompt(self, extracted_data: Dict) -> str:
        return self.generate_structured_description(extracted_data)
