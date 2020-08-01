"""
TODO: more testing

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random


class DummyBlock:

    def __init__(self):
        self.idx = 0.0

    def __call__(self, x):
        self.idx += 1.0
        return torch.tensor(self.idx)


class FractalBlock(nn.Module):
    def __init__(self, num_zooms, block=DummyBlock, residual=False, prob=.9):
        super().__init__()
        self.prob = prob
        self.residual = residual
        self.block = block()
        self.num_zooms = num_zooms
        if num_zooms <= 0:
            self.zoom = None
        else:
            self.zoom = FractalBlock(num_zooms-1, block=block)
        self.outputs = []

    def drop_path(self, xs):
        # make sure there is at least 1 prediction to average
        keep_idx = random.choice(range(len(xs)))
        keep = xs[keep_idx]
        # keep each remaining candidate with probability self.prob
        candidates = xs[:keep_idx]+xs[keep_idx+1:]
        return [keep] + [k for k in candidates if random.uniform(0, 1) < self.prob]

    def collect_outputs(self):
        if self.zoom is None:
            outputs = [torch.stack(self.outputs)]
        else:
            outputs = self.zoom.collect_outputs() + [torch.stack(self.outputs)]
        self.outputs = []
        return outputs

    def forward(self, x):
        """

        :param x: bs X dim
        :return: x_agg list of bs X dim tensors which has len num_zooms + 1
        """
        z_agg = []
        if self.residual:
            x_next = x + self.block(x)
        else:
            x_next = self.block(x)
        self.outputs.append(x_next)
        if self.zoom is not None:
            z_agg = self.zoom(x)
            z_next = torch.mean(torch.stack(self.drop_path(z_agg)), dim=0)
            z_agg = self.zoom(z_next)
        return [x_next] + z_agg


class RecurrentFractal(nn.Module):

    def __init__(self, num_zooms, block=DummyBlock, residual=True, prob=.9):
        super().__init__()
        self.residual = residual
        self.num_zooms = num_zooms
        self.step = FractalBlock(num_zooms, block=block, residual=residual, prob=prob)

    def calculate_loss(self, Xtrue):
        predictions = self.step.collect_outputs()
        losses = []
        for idx, pred in enumerate(predictions):
            xtrue = Xtrue[0::2**idx]
            losses.append(F.mse_loss(xtrue, pred))
        return predictions, torch.mean(torch.stack(losses))

    def forward(self, x0, Xtrue):
        """

        :param X: (num_steps X bs X dim)
        :return: preds: a list of tensors shapes=(num_steps/2^numzooms, bs, dim), (numsteps/2^(numzooms-1), bs, dim), ... (numsteps, bs, dim)
                 loss: a scalar tensor
        """
        for i in range(len(Xtrue[::2**self.num_zooms])):
            if self.residual:
                x0 += torch.mean(torch.stack(self.step(x0), dim=0))
            else:
                x0 = torch.mean(torch.stack(self.step(x0)), dim=0)
        return self.calculate_loss(Xtrue)


if __name__ == '__main__':
    model = FractalBlock(3)
    x_out = model(torch.tensor(1.0))
    model = RecurrentFractal(3, prob=1.0)
    print(model(torch.tensor(1.0), torch.tensor(range(1, 9), dtype=torch.float32)))

