import random

import torch
import numpy as np

def set_random_seed(seed):
    if seed is not None and seed > 0:
        random.seed(seed)
        np.random(seed)
        torch.manual_seed(seed)