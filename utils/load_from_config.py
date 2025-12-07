import yaml

def load_from_config(cfg_path):

    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
        
    return cfg