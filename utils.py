import random
import os
from glob import glob
import numpy as np
import yaml
from datetime import datetime
import torch

def current_timestamp():
    return datetime.now().strftime('%y-%m-%d-%H-%M-%S')

def load_config(config_file):
    with open(config_file, 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def get_last_checkpoint(checkpoint_dir, return_best=False):
    if return_best:
        return [x for x in sorted(glob(f"{checkpoint_dir}/*.ckpt"), key=os.path.getmtime) if 'last' not in x][-1]
    return f"{checkpoint_dir}/last.ckpt"

def fix_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False