from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def config_path(fn):
    """
    Construct absolute Path object pointing to a configuration file inside the project
    `config/` directory.

    Args:
        fn (str | Path): Filename or relative path inside the configuration directory.

    Returns:
        Path: Resolved absolute path to target configuration file.
    """
    dir_path = PROJECT_ROOT / "config"
    dir_path.mkdir(parents=True, exist_ok=True)
    full_path = PROJECT_ROOT / "config" / fn
    return full_path

def data_path(fn):
    """
    Construct absolute Path object pointing to a data file inside the project `data/`
    directory.

    Args:
        fn (str | Path): Filename or relative path inside the data directory.

    Returns:
        Path: Resolved absolute path to target data file.
    """
    dir_path = PROJECT_ROOT / "data"
    dir_path.mkdir(parents=True, exist_ok=True)
    full_path = PROJECT_ROOT / "data" / fn
    return full_path

def models_path(dir_name, fn):
    """
    Construct absolute Path object pointing to a model artifact file inside a
    subdirectory of `models/`.

    Args:
        dir_name (str | Path): Subdirectory name inside models directory.
        fn (str | Path): Target model filename.

    Returns:
        Path: Resolved absolute path to target model file.
    """
    dir_path = PROJECT_ROOT / "models" / dir_name
    dir_path.mkdir(parents=True, exist_ok=True)
    full_path = PROJECT_ROOT / "models" / dir_name / fn
    return full_path

def results_path(dir_name, fn):
    """
    Construct absolute Path object pointing to a results output file inside a
    subdirectory of `results/`.

    Args:
        dir_name (str | Path): Subdirectory name inside results directory.
        fn (str | Path): Target results filename.

    Returns:
        Path: Resolved absolute path to target results file.
    """
    dir_path = PROJECT_ROOT / "results" / dir_name
    dir_path.mkdir(parents=True, exist_ok=True)
    full_path = PROJECT_ROOT / "results" / dir_name / fn
    return full_path