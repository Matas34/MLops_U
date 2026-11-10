import os

import yaml


def load_params():
    params = {}

    try:
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
    except FileNotFoundError:
        env_map = {
            "DEGREE":    ("degree",    int),
            "DATA_SIZE": ("data_size", int),
            "SEED":      ("seed",      int),
            "HIDDEN":    ("hidden",    int),
            "LR":        ("lr",        float),
            "ALPHA":     ("alpha",     float),
            "EPOCHS":    ("epochs",    int),
        }
        
        for env_name, (key, cast) in env_map.items():
            if env_name in os.environ:
                params[key] = cast(os.environ[env_name])

    return params