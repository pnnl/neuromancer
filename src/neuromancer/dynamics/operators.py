from typing import TypeVar

import torch
from torch import nn

from neuromancer.modules.blocks import Block

TDeepONet = TypeVar("TDeepONet", bound="DeepONet")

# TODO(Colby): abstract class to inherit from?

class DeepONet(nn.Module):
    """Deep Operator Network."""

    def __init__(
            self: TDeepONet,
            branch_net: Block,
            trunk_net: Block,
            bias=True,
    ) -> None:
        """: param insize_branch: (int) dimensionality of branch net input
        : param insize_trunk: (int) dimensionality of trunk net input
        : param widthsize: (int) horizontal size for branch and trunk net
        : param interactsize: (int) dimensionality of branch net and trunk net output
        : param depth_branch: (int) depth of branch net
        : param depth_trunk: (int) depth of trunk net
        : nonlin: (callable) nonlinear activation function for branch net and trunk net
        : bias: (bool) Whether to use bias
        """
        super().__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=not bias)

    def transpose_branch_inputs(self, branch_inputs):
        transposed_branch_inputs = torch.transpose(branch_inputs, 0, 1)
        return transposed_branch_inputs


    def reg_error(self: TDeepONet):
        return self.branch_net.reg_error() + self.trunk_net.reg_error()

    def forward(self, branch_inputs: torch.Tensor, trunk_inputs: torch.Tensor) -> torch.Tensor:
        """TODO param types and descriptions
        """
        branch_output = self.branch_net(self.transpose_branch_inputs(branch_inputs))
        trunk_output = self.trunk_net(trunk_inputs)
        result = torch.matmul(branch_output, trunk_output.T) + self.bias
        # TODO, return branch_output and trunk_output for control
        return result
