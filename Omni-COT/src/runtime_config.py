import os
from typing import Dict, Tuple

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()


def _resolve_env_placeholder(value: str) -> str:
    if not value:
        return ''
    value = value.strip()
    if value.startswith('${') and value.endswith('}'):
        env_key = value[2:-1].strip()
        return os.getenv(env_key, '')
    return value


def _get_provider_config(config: Dict) -> Dict:
    provider = config.get('provider')
    if isinstance(provider, dict):
        return provider
    legacy_provider = config.get('aliyun')
    if isinstance(legacy_provider, dict):
        return legacy_provider
    return {}


def get_api_settings(config: Dict) -> Tuple[str, str]:
    provider = _get_provider_config(config)
    api_key = _resolve_env_placeholder(provider.get('api_key', ''))
    base_url = _resolve_env_placeholder(provider.get('base_url', ''))

    if not api_key:
        api_key = os.getenv('OPENAI_API_KEY', '')
    if not base_url:
        base_url = os.getenv('OPENAI_BASE_URL', '')

    missing = []
    if not api_key:
        missing.append('OPENAI_API_KEY')
    if not base_url:
        missing.append('OPENAI_BASE_URL')
    if missing:
        raise ValueError(f"Missing required API settings: {', '.join(missing)}")

    return api_key, base_url


def get_model_settings(config: Dict) -> Dict[str, str]:
    provider = _get_provider_config(config)
    models = provider.get('models', {})
    if not isinstance(models, dict):
        models = {}

    model_settings = {
        'vision': _resolve_env_placeholder(str(models.get('vision', '')).strip()),
        'reasoning': _resolve_env_placeholder(str(models.get('reasoning', '')).strip()),
        'text': _resolve_env_placeholder(str(models.get('text', '')).strip()),
    }

    missing = [name for name, value in model_settings.items() if not value]
    if missing:
        raise ValueError(f"Missing required model settings: {', '.join(missing)}")

    return model_settings
