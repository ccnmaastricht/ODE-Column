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

def load_pkl_file(fn):
    """
    Load and deserialize python objects from a binary pickle (`.pkl`) file.

    Args:
        fn (str | Path): Path to target `.pkl` file.

    Returns:
        Any: Deserialized python object payload.
    """
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    return data

def save_pkl_file(fn, data):
    """
    Serialize and save python data objects to a binary pickle (`.pkl`) file.

    Args:
        fn (str | Path): Output path for target `.pkl` file.
        data (Any): Python object to serialize.
    """
    with open(fn, 'wb') as  f:
        pickle.dump(data, f)
