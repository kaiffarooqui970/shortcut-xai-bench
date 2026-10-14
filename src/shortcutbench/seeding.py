"""Seed every RNG we touch and log what was set."""
from __future__ import annotations

import logging
import os
import random

import numpy as np
import torch

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    log.info("seeded python/numpy/torch with %d", seed)
