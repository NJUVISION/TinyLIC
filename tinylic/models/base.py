import math
import warnings

from typing import cast

import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.models.utils import update_registered_buffers

__all__ = [
    "CompressionModel",
    "SimpleVAECompressionModel",
    "get_scale_table",
    "SCALES_MIN",
    "SCALES_MAX",
    "SCALES_LEVELS",
]


SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))


class CompressionModel(nn.Module):
    def __init__(self, entropy_bottleneck_channels=None, init_weights=None):
        super().__init__()

        if entropy_bottleneck_channels is not None:
            warnings.warn(
                "The entropy_bottleneck_channels parameter is deprecated. "
                "Create an entropy_bottleneck in your model directly instead:\n\n"
                "class YourModel(CompressionModel):\n"
                "    def __init__(self):\n"
                "        super().__init__()\n"
                "        self.entropy_bottleneck = "
                "EntropyBottleneck(entropy_bottleneck_channels)\n",
                DeprecationWarning,
                stacklevel=2,
            )
            self.entropy_bottleneck = EntropyBottleneck(entropy_bottleneck_channels)

        if init_weights is not None:
            warnings.warn(
                "The init_weights parameter was removed as it was never functional.",
                DeprecationWarning,
                stacklevel=2,
            )

    def load_state_dict(self, state_dict, strict=True):
        for name, module in self.named_modules():
            if not any(x.startswith(name) for x in state_dict.keys()):
                continue

            if isinstance(module, EntropyBottleneck):
                update_registered_buffers(
                    module,
                    name,
                    ["_quantized_cdf", "_offset", "_cdf_length"],
                    state_dict,
                )

            if isinstance(module, GaussianConditional):
                update_registered_buffers(
                    module,
                    name,
                    ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
                    state_dict,
                )

        return nn.Module.load_state_dict(self, state_dict, strict=strict)

    def update(self, scale_table=None, force=False):
        """Updates EntropyBottleneck and GaussianConditional CDFs.

        Needs to be called once after training to be able to later perform the
        evaluation with an actual entropy coder.

        Args:
            scale_table (torch.Tensor): table of scales (i.e. stdev)
                for initializing the Gaussian distributions
                (default: 64 logarithmically spaced scales from 0.11 to 256)
            force (bool): overwrite previous values (default: False)
            update_quantiles (bool): fast update quantiles (default: False)

        Returns:
            updated (bool): True if at least one of the modules was updated.
        """
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)

        updated = False
        for m in self.children():
            if not isinstance(m, EntropyBottleneck):
                continue
            rv = m.update(force=force)
            updated |= rv
        return updated

    def aux_loss(self) -> Tensor:
        loss = sum(m.loss() for m in self.modules() if isinstance(m, EntropyBottleneck))
        return cast(Tensor, loss)
