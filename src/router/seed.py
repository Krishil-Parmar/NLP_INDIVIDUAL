"""Random seeds and device choice, shared by every training script and the notebook."""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every random number generator the project uses.

    torch.manual_seed also seeds the GPU generators (CUDA and Apple MPS).
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> torch.device:
    """Fastest available device: NVIDIA GPU, then Apple GPU (MPS), then CPU.

    Setting the environment variable ROUTER_DEVICE (e.g. ROUTER_DEVICE=cpu)
    overrides this, which is used to test how the notebook runs on a CPU-only machine.
    """
    if os.environ.get("ROUTER_DEVICE"):
        return torch.device(os.environ["ROUTER_DEVICE"])
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
