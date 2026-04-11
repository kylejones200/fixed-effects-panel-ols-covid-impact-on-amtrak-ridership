import pandas as pd


def read_csv(path_or_url: str, **kwargs) -> pd.DataFrame:
    """Load a CSV from a local path or URL."""
    return pd.read_csv(path_or_url, **kwargs)
