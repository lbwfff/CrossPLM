import os
from datetime import datetime


def create_task_folder(base_dir: str, task_name: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{task_name}_{timestamp}"
    task_dir = os.path.join(base_dir, folder_name)
    os.makedirs(task_dir, exist_ok=True)
    return task_dir


def save_config(config, path: str):
    if hasattr(config, "to_yaml"):
        config.to_yaml(path)
    else:
        import yaml
        with open(path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def load_config(path: str, config_class=None):
    if config_class is not None:
        return config_class.from_yaml(path)
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)
