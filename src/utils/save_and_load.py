import tomllib
import pickle


def load_config(fn):
    """
    Load and deserialize configuration parameters from a TOML file.

    Args:
        fn (str | Path): Path to target `.toml` configuration file.

    Returns:
        dict: Parsed configuration parameters dictionary.
    """
    with open(fn, 'rb') as f:
        return tomllib.load(f)
