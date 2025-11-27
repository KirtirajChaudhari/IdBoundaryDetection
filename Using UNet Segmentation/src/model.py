"""
Model factory.

- Prefers segmentation_models_pytorch (SMP) U-Net with a lightweight ResNet-18 encoder
  for faster training and lower memory on GPUs like GTX 1650.
- Falls back to a small custom UNet if SMP is not available.

- Supports optional gradient checkpointing:
  * For SMP encoders that expose `set_grad_checkpointing()` (timm-backed encoders).
  * For the fallback UNetSmall by using torch.utils.checkpoint on the deeper blocks.
"""

import torch
import torch.nn as nn
from typing import Optional


def build_smp_unet(encoder: str = "resnet18",
                   encoder_weights: Optional[str] = "imagenet",
                   use_grad_checkpointing: bool = False):
    """
    Tries to build an SMP U-Net with the given encoder.
    If gradient checkpointing is requested and the encoder supports
    `set_grad_checkpointing()`, it is enabled.
    """
    try:
        import segmentation_models_pytorch as smp
    except Exception:
        return None

    model = smp.Unet(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,
    )

    # Enable grad checkpointing on encoder if supported
    if use_grad_checkpointing:
        encoder_mod = getattr(model, "encoder", None)
        # Many timm-based encoders expose set_grad_checkpointing
        if encoder_mod is not None and hasattr(encoder_mod, "set_grad_checkpointing"):
            try:
                encoder_mod.set_grad_checkpointing()
            except Exception:
                # If not supported, ignore gracefully
                pass

    return model


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNetSmall(nn.Module):
    """
    Minimal UNet-style architecture as a fallback when SMP is not available.

    Gradient checkpointing is optionally enabled for the deeper blocks to
    reduce memory usage at the cost of extra compute.
    """

    def __init__(self, in_ch: int = 3, out_ch: int = 1, use_grad_checkpointing: bool = False):
        super().__init__()
        self.use_grad_checkpointing = use_grad_checkpointing

        self.down1 = DoubleConv(in_ch, 32)
        self.pool = nn.MaxPool2d(2)
        self.down2 = DoubleConv(32, 64)
        self.down3 = DoubleConv(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.conv_up1 = DoubleConv(64, 32)

        self.outc = nn.Conv2d(32, out_ch, kernel_size=1)

    def _forward_block_with_checkpoint(self, block: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """
        Applies checkpointing to a block if enabled and gradients are required.
        """
        from torch.utils.checkpoint import checkpoint

        if (self.use_grad_checkpointing
                and self.training
                and x.requires_grad
                and torch.cuda.is_available()):
            return checkpoint(block, x)
        return block(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        c1 = self._forward_block_with_checkpoint(self.down1, x)
        p1 = self.pool(c1)

        c2 = self._forward_block_with_checkpoint(self.down2, p1)
        p2 = self.pool(c2)

        c3 = self._forward_block_with_checkpoint(self.down3, p2)

        # Decoder
        u2 = self.up2(c3)
        u2 = torch.cat([u2, c2], dim=1)
        c_up2 = self._forward_block_with_checkpoint(self.conv_up2, u2)

        u1 = self.up1(c_up2)
        u1 = torch.cat([u1, c1], dim=1)
        c_up1 = self._forward_block_with_checkpoint(self.conv_up1, u1)

        out = self.outc(c_up1)
        return out


def build_model(
    preferred: str = "smp",
    encoder: str = "resnet18",
    encoder_weights: Optional[str] = "imagenet",
    use_grad_checkpointing: bool = False,
) -> nn.Module:
    """
    Factory to build the segmentation model.

    Args:
        preferred: "smp" (default) to use segmentation_models_pytorch if available.
        encoder: encoder name for SMP (defaults to resnet18 for lightweight speed).
        encoder_weights: typically "imagenet" or None.
        use_grad_checkpointing: enable gradient checkpointing where supported.

    Returns:
        nn.Module instance ready for training/inference.
    """
    if preferred == "smp":
        model = build_smp_unet(
            encoder=encoder,
            encoder_weights=encoder_weights,
            use_grad_checkpointing=use_grad_checkpointing,
        )
        if model is not None:
            return model

    # Fallback: small UNet with optional gradient checkpointing
    return UNetSmall(in_ch=3, out_ch=1, use_grad_checkpointing=use_grad_checkpointing)
