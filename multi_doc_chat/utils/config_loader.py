from pathlib import Path
import yaml
import os



def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file and return its contents as a dictionary.

    Args:
        config_path (str): The path to the YAML configuration file"""
    
    path=Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(path, 'r', encoding='utf-8') as file:
        try:
            config = yaml.safe_load(file)
            if config is None:
                raise ValueError(f"Configuration file is empty: {config_path}")
            return config
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")
        
