"""ConvLSTM forecasters (models A-D), the symmetrised wrapper and the two reference predictors."""
import torch
from torch import nn
import torch.nn.functional as F

from . import transforms

PATCH = 4
HIDDEN = 64
STEPS = 9  # 5 input frames + 4 fed-back predictions


class ConvLSTMCell(nn.Module):
    """ConvLSTM without peephole terms: one convolution makes all four gates."""

    def __init__(self, in_channels, hidden, kernel=3, dilation=1):
        super().__init__()
        self.hidden = hidden
        self.conv = nn.Conv2d(in_channels + hidden, 4 * hidden, kernel,
                              padding=dilation * (kernel // 2), dilation=dilation, bias=True)

    def forward(self, u, h, c):
        i, f, o, g = self.conv(torch.cat([u, h], dim=1)).chunk(4, dim=1)
        c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h = torch.sigmoid(o) * torch.tanh(c)
        return h, c


class ConvLSTMForecaster(nn.Module):
    """Pixel-unshuffle -> 2 ConvLSTM layers -> 1x1 conv -> pixel-shuffle -> sigmoid.

    Model A uses dilations (1, 1). Model D uses (1, 2): the layer-2 convolution is
    dilated, which widens its reach with no extra parameters.
    """

    def __init__(self, dilations=(1, 1)):
        super().__init__()
        channels = PATCH * PATCH
        self.layer1 = ConvLSTMCell(channels, HIDDEN, dilation=dilations[0])
        self.layer2 = ConvLSTMCell(HIDDEN, HIDDEN, dilation=dilations[1])
        self.output = nn.Conv2d(HIDDEN, channels, 1, bias=True)

    def forward(self, x):
        """x: [B, 5, 1, 64, 64] input frames -> [B, 5, 1, 64, 64] predicted frames 6-10.

        Steps 1-5 read the input frames, steps 6-9 read the network's own previous
        prediction (no teacher forcing). Outputs of steps 5-9 are the forecast.
        """
        b, _, _, height, width = x.shape
        size = (b, HIDDEN, height // PATCH, width // PATCH)
        h1 = c1 = h2 = c2 = x.new_zeros(size)
        outputs = []
        prediction = None
        for step in range(STEPS):
            frame = x[:, step] if step < x.shape[1] else prediction
            u = F.pixel_unshuffle(frame, PATCH)
            h1, c1 = self.layer1(u, h1, c1)
            h2, c2 = self.layer2(h1, h2, c2)
            prediction = torch.sigmoid(F.pixel_shuffle(self.output(h2), PATCH))
            if step >= x.shape[1] - 1:
                outputs.append(prediction)
        return torch.stack(outputs, dim=1)


class Symmetrised(nn.Module):
    """f_bar(X) = 1/8 sum_g g^{-1}(f(g(X))): transform, predict, undo, average."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        b = x.shape[0]
        stacked = torch.cat([transforms.apply(x, g) for g in transforms.GROUP], dim=0)
        predictions = self.model(stacked).split(b, dim=0)
        undone = [transforms.invert(p, g) for p, g in zip(predictions, transforms.GROUP)]
        return torch.stack(undone, dim=0).mean(dim=0)


class Blank(nn.Module):
    """Predicts all-zero frames."""

    def forward(self, x):
        return torch.zeros_like(x)


class CopyLast(nn.Module):
    """Repeats the last input frame five times."""

    def forward(self, x):
        return x[:, -1:].expand(-1, 5, -1, -1, -1).clone()


# Model name -> (dilations, speed augmentation, symmetry augmentation).
CONFIGS = {
    "A": ((1, 1), False, False),
    "B": ((1, 1), False, True),
    "C": ((1, 1), True, False),
    "D": ((1, 2), True, False),
    "DB": ((1, 2), True, True),
}


def build(name):
    dilations, _, _ = CONFIGS[name]
    return ConvLSTMForecaster(dilations)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
