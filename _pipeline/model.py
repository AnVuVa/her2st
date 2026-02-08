from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))


class Up(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        diff_y = skip.size(2) - x.size(2)
        diff_x = skip.size(3) - x.size(3)
        if diff_y != 0 or diff_x != 0:
            x = nn.functional.pad(
                x,
                [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2],
            )

        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class SimpleUNet(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 32) -> None:
        super().__init__()
        c1 = base_channels
        c2 = c1 * 2
        c3 = c2 * 2
        c4 = c3 * 2

        self.inc = DoubleConv(in_channels, c1)
        self.down1 = Down(c1, c2)
        self.down2 = Down(c2, c3)
        self.down3 = Down(c3, c4)

        self.up1 = Up(c4, c3, c3)
        self.up2 = Up(c3, c2, c2)
        self.up3 = Up(c2, c1, c1)

        self.out_conv = nn.Conv2d(c1, c1, kernel_size=1)
        self.out_channels = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.out_conv(x)
        return x


class AModel(nn.Module):
    """Simple U-Net regressor for spot-level gene prediction."""

    def __init__(
        self,
        out_dim: int,
        in_channels: int = 3,
        base_channels: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone = SimpleUNet(in_channels=in_channels, base_channels=base_channels)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(self.backbone.out_channels, out_dim),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        feats = self.backbone(image)
        pooled = self.pool(feats)
        pred = self.head(pooled)
        return pred

    @staticmethod
    def loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return nn.functional.mse_loss(pred, target)

    def training_step(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        pred = self.forward(batch["image"])
        loss = self.loss_fn(pred, batch["target_gene"])
        metrics = {"mse": float(loss.detach().item())}
        return loss, metrics

    # Model information report
    def model_report(self) -> Dict[str, object]:
        report = {
            "name": self.__class__.__name__,
            "num_parameters": sum(p.numel() for p in self.parameters() if p.requires_grad),
        }
        return report


if __name__ == "__main__":
    model = AModel(out_dim=785)
    x = torch.randn(2, 3, 112, 112)
    out = model(x)
    print("Output shape:", tuple(out.shape))
    print(model)
    print(model.model_report())
