from typing import Dict, Tuple


class SceneUnderstanding:
    def __init__(self, config: dict):
        self.config = config

    def understand_scene_from_json(self, scene_data_text: str) -> Tuple[str, Dict]:
        usage_info = {
            'prompt_tokens': 0,
            'completion_tokens': 0,
            'total_tokens': 0,
        }
        return scene_data_text, usage_info
