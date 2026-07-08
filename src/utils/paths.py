from pathlib import Path



PROJECT_ROOT = Path(__file__).resolve().parents[2]



def config_dir(fn):
    dir_path = PROJECT_ROOT / "config"
    dir_path.mkdir(parents=True, exist_ok=True)
    full_path = PROJECT_ROOT / "config" / fn
    return full_path

def data_dir(fn):
    dir_path = PROJECT_ROOT / "data"
    dir_path.mkdir(parents=True, exist_ok=True)
    full_path = PROJECT_ROOT / "data" / fn
    return full_path

def models_dir(dir_name, fn):
    dir_path = PROJECT_ROOT / "models" / dir_name
    dir_path.mkdir(parents=True, exist_ok=True)
    full_path = PROJECT_ROOT / "models" / dir_name / fn
    return full_path

def results_dir(dir_name, fn):
    dir_path = PROJECT_ROOT / "results" / dir_name
    dir_path.mkdir(parents=True, exist_ok=True)
    full_path = PROJECT_ROOT / "results" / dir_name / fn
    return full_path