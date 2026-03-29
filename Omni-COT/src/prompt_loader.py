from pathlib import Path
from string import Template
from functools import lru_cache
from typing import Optional


class PromptLoader:
    def __init__(self, prompt_dir: Optional[str] = None):
        if prompt_dir:
            self.prompt_dir = Path(prompt_dir)
        else:
            self.prompt_dir = Path(__file__).resolve().parents[1] / 'prompts'

    @lru_cache(maxsize=256)
    def load(self, relative_path: str) -> str:
        prompt_path = self.prompt_dir / relative_path
        return prompt_path.read_text(encoding='utf-8')

    def render(self, relative_path: str, **kwargs) -> str:
        template = Template(self.load(relative_path))
        rendered_kwargs = {key: '' if value is None else str(value) for key, value in kwargs.items()}
        return template.safe_substitute(rendered_kwargs)
