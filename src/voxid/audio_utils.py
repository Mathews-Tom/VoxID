from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    import torch


def resample_linear(
    audio: npt.NDArray[np.float32],
    target_length: int,
) -> npt.NDArray[np.float32]:
    """Resample a 1-D audio array by linear interpolation.

    Args:
        audio: Input audio array.
        target_length: Number of samples in the output.

    Returns:
        Resampled array with shape (target_length,), dtype float32.
    """
    indices = np.linspace(0, len(audio) - 1, target_length)
    result: npt.NDArray[np.float32] = np.interp(
        indices, np.arange(len(audio)), audio
    ).astype(np.float32)
    return result


def resample_linear_torch(
    tensor: torch.Tensor,
    target_length: int,
) -> torch.Tensor:
    """Resample a 1-D torch tensor by linear interpolation.

    Args:
        tensor: Input 1-D tensor.
        target_length: Number of samples in the output.

    Returns:
        Resampled tensor with dtype float32.
    """
    import torch as _torch

    indices = _torch.linspace(0, len(tensor) - 1, target_length)
    result: torch.Tensor = _torch.from_numpy(
        np.interp(
            indices.numpy(), np.arange(len(tensor)), tensor.numpy()
        ).astype(np.float32)
    )
    return result
