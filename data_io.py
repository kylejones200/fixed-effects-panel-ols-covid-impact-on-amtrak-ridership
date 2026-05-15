import pandas as pd



from pathlib import Path


def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    if not config_path.exists():
        return {}
    with open(config_path) as _f:
        import yaml as _yaml
        return _yaml.safe_load(_f) or {}

def read_csv(path_or_url: str, **kwargs) -> pd.DataFrame:
    """Load a CSV from a local path or URL."""
    return pd.read_csv(path_or_url, **kwargs)
