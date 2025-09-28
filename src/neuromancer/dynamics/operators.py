from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar

import torch
from torch import nn

if TYPE_CHECKING:
    from neuromancer.modules.blocks import Block

TDeepONet = TypeVar("TDeepONet", bound="DeepONet")


class DeepONet(nn.Module):
    """Deep Operator Network."""

    def __init__(
            self: TDeepONet,
            branch_net: Block,
            trunk_net: Block,
            bias: bool = True,
    ) -> None:
        """Deep Operator Network.

        :param branch_net: (Block) Branch network
        :param trunk_net: (Block) Trunk network
        :param bias: (bool) Whether to use bias or not
        """
        super().__init__()
        self.branch_net = branch_net
        self.trunk_net = trunk_net
        self.bias = nn.Parameter(torch.zeros(1), requires_grad=not bias)

    @staticmethod
    def transpose_branch_inputs(branch_inputs: torch.Tensor) -> torch.Tensor:
        """Transpose branch inputs.

        :param branch_inputs: (torch.Tensor, shape=[Nu, Nsamples])
        :return: (torch.Tensor, shape=[Nsamples, Nu])
        """
        transposed_branch_inputs = torch.transpose(branch_inputs, 0, 1)
        return transposed_branch_inputs

    def forward(self: TDeepONet, branch_inputs: torch.Tensor,
                trunk_inputs: torch.Tensor) -> tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward propagation.
        Nsamples = should be batch size, but if total/batch size isn't even then what will the behavior be, is batch size respected
        Nu = number of sensors
        in_size_trunk = 1, why?
        interact_size = out size and interact size for both networks, why

        :param branch_inputs: (torch.Tensor, shape=[Nu, Nsamples])
        :param trunk_inputs: (torch.Tensor, shape=[Nu, in_size_trunk])
        :return:
            output: (torch.Tensor, shape=[Nsamples, Nu]),
            branch_output: (torch.Tensor, shape=[Nsamples, interact_size]),
            trunk_output: (torch.Tensor, shape=[Nu, interact_size])
        """
        branch_output = self.branch_net(self.transpose_branch_inputs(branch_inputs))
        trunk_output = self.trunk_net(trunk_inputs)
        output = torch.matmul(branch_output, trunk_output.T) + self.bias
        # return branch_output and trunk_output as well for control use cases
        return output, branch_output, trunk_output
