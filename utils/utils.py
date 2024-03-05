"""Contains utility functions."""

import importlib
from typing import Any
import torch
import random
import numpy as np
import os


def load_obj(obj_path: str, default_obj_path: str = "") -> Any:
    """
    Extract an object from a given path.

    Args:
        obj_path - Path to an object to be extracted, including the object name.
        default_obj_path - Default object path.
    Returns:
        Extracted object.
    Raises:
        AttributeError: When the object does not have the given named attribute.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
    """
    obj_path_list = obj_path.rsplit(".", 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError("Object `{}` cannot be loaded from `{}`.".format(obj_name, obj_path))
    return getattr(module_obj, obj_name)


def seed_everything(seed=42):
    """Fix random seed for experiments."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
