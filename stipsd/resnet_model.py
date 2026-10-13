#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ResNet model adapted from the original S1_ALL regression script."""
from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):  # type: ignore[no-untyped-def]
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class ResNetRegression(nn.Module):
    def __init__(
        self,
        layers: Sequence[int] = (2, 2, 2),
        channels: Sequence[int] = (16, 32, 64),
        output_dim: int = 3,
    ) -> None:
        super().__init__()
        self.in_channels = channels[0]
        self.conv1 = nn.Conv2d(1, channels[0], kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(channels[0], layers[0], stride=1)
        self.layer2 = self._make_layer(channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(channels[2], layers[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channels[2], output_dim)

    def _make_layer(self, out_channels: int, blocks: int, stride: int = 1) -> nn.Sequential:
        layers = [BasicBlock(self.in_channels, out_channels, stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):  # type: ignore[no-untyped-def]
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def create_model(model_width: str = "standard", output_dim: int = 3) -> ResNetRegression:
    if model_width == "standard":
        channels = (16, 32, 64)
    elif model_width == "wide":
        channels = (32, 64, 128)
    else:
        raise ValueError(f"Unknown model_width: {model_width}")
    return ResNetRegression(layers=(2, 2, 2), channels=channels, output_dim=output_dim)

