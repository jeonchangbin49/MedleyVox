# For replacing the original asteroid_filterbanks transforms.py

import torch
from typing import Tuple
from asteroid_filterbanks.transforms import mag, check_complex


# Added EPS
def angle(tensor, dim: int = -2, EPS=1e-8):
    """Return the angle of the complex-like torch tensor.
    Args:
        tensor (torch.Tensor): the complex tensor from which to extract the
            phase.
        dim(int, optional): the frequency (or equivalent) dimension along which
            real and imaginary values are concatenated.
    Returns:
        :class:`torch.Tensor`:
            The counterclockwise angle from the positive real axis on
            the complex plane in radians.
    """
    check_complex(tensor, dim=dim)
    real, imag = torch.chunk(tensor, 2, dim=dim)
    return torch.atan2(
        imag, real + EPS
    )  # Let's consider epsilon here. This can prevent the NaN behavior while calculating atan2.


# No modification
def my_magphase(spec: torch.Tensor, dim: int = -2) -> Tuple[torch.Tensor, torch.Tensor]:
    """Splits Asteroid complex-like tensor into magnitude and phase."""
    mag_val = mag(spec, dim=dim)
    phase = angle(spec, dim=dim)
    return mag_val, phase
