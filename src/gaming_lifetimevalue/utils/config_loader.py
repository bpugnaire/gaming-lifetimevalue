from pathlib import Path
import yaml

def load_config(config_path: str = "confs/params.yml") -> dict:
    """Load and parse configuration file"""
    with open(Path(config_path), 'r') as f:
        params = yaml.safe_load(f)
    
    return params