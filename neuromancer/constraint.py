"""
Definition of neuromancer.Constraint class used in conjunction with neuromancer.variable class.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuromancer.variable import Variable


class LT(nn.Module):
    """Less than constraint for upper bounding the left hand side by the right hand side."""
    def __init__(self, norm=1):
        super().__init__()
        self.norm = norm

    def forward(self, left, right):
        if self.norm == 1:
            return torch.mean(F.relu(left - right))
        elif self.norm == 2:
            return torch.mean((F.relu(left - right))**2)


class GT(nn.Module):
    """Greater than constraint for lower bounding the left hand side by the right hand side."""

    def __init__(self, norm=1):
        super().__init__()
        self.norm = norm

    def forward(self, left, right):
        if self.norm == 1:
            return torch.mean(F.relu(right - left))
        elif self.norm == 2:
            return torch.mean((F.relu(right - left)) ** 2)


class Eq(nn.Module):
    """Equality constraint penalizing difference between left and right hand side."""
    def __init__(self, norm=1):
        super().__init__()
        self.norm = norm

    def forward(self, left, right):
        if self.norm == 1:
            return F.l1_loss(left, right)
        elif self.norm == 2:
            return F.mse_loss(left, right)


class Constraint(nn.Module):

    def __init__(self, left, right, comparator, weight=1.0):
        """

        :param left: (nm.Variable or numeric) Left hand side of equality or inequality constraint
        :param right: (nm.Variable or numeric) Right hand side of equality or inequality constraint
        :param comparator: (nn.Module) Intended to be LE, GE, LT, GT, or Eq object, but can be any nn.Module
                                       which satisfies the Comparator interface (init function takes an integer norm and
                                       object has an integer valued self.norm attribute.
        """
        super().__init__()
        if not type(left) is Variable:
            left = Variable(str(left), constant=left)
        if not type(right) is Variable:
            right = Variable(str(right), constant=right)
        self.left = left
        self.right = right
        self.comparator = comparator
        self.weight = weight

    @property
    def variable_names(self):
        return [self.left.key, self.right.key]

    def __xor__(self, norm):
        comparator = type(self.comparator)(norm=norm)
        return Constraint(self.left, self.right, comparator, weight=self.weight)

    def __mul__(self, weight):
        return Constraint(self.left, self.right, self.comparator, weight=weight)

    def __rmul__(self, weight):
        return Constraint(self.left, self.right, self.comparator, weight=weight)

    def forward(self, variables):
        return self.weight*self.comparator(self.left(variables), self.right(variables))