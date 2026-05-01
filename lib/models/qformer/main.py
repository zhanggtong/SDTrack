import math
from functools import partial, wraps
from math import ceil

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from PIL import Image
from torchvision import transforms as T


def exists(val):
    """
    Check if the value is not None.

    Args:
        val: The value to check.

    Returns:
        bool: True if value exists (is not None), False otherwise.
    """
    return val is not None


def default(val, d):
    """
    Return the value if it exists, otherwise return a default value.

    Args:
        val: The value to check.
        d: The default value to return if val is None.

    Returns:
        The value if it exists, otherwise the default value.
    """
    return val if exists(val) else d




def l2norm(t, groups=1):
    t = rearrange(t, "... (g d) -> ... g d", g=groups)
    t = F.normalize(t, p=2, dim=-1)
    return rearrange(t, "... g d -> ... (g d)")


