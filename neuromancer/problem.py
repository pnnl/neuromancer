"""

"""
# python base imports
from typing import Dict, List, Callable

# machine learning/data science imports
import torch
import torch.nn as nn


class Objective(nn.Module):
    def __init__(self, variable_names: List[str], loss: Callable[..., torch.Tensor], weight=1.0, name='objective'):
        """

        :param variable_names: List of str
        :param loss: (callable) Number of arguments of the callable should equal the number of strings in variable names.
                                Arguments to callable should be torch.Tensor and return type a 0-dimensional torch.Tensor
        :param weight: (float) Weight of objective for calculating multi-objective loss function
        :param name: (str) Name for tracking output
        """
        super().__init__()
        self.variable_names = variable_names
        self.weight = weight
        self.loss = loss
        self.name = name

    def forward(self, variables: Dict[str, torch.Tensor]) -> torch.Tensor:
        """

        :param variables: (dict, {str: torch.Tensor}) Should contain keys corresponding to self.variable_names
        :return: 0-dimensional torch.Tensor that can be cast as a floating point number
        """
        return self.weight*self.loss(*[variables[k] for k in self.variable_names])


class Problem(nn.Module):

    def __init__(self, objectives: List[Objective], constraints: List[Objective],
                 components: List[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]]):
        """
        This is similar in spirit to a nn.Sequential module. However,
        by concatenating input and output dictionaries for each component
        module we can represent arbitrary directed acyclic computation graphs.
        In addition the Problem module takes care of calculating weighted multi-objective
        loss functions via the lists of Objective objects (constraints and objectives) which calculate loss terms
        from aggregated input and set of outputs from the component modules.

        :param objectives: list of Objective objects
        :param constraints: list of Objective objects
        :param components: list of objects which implement the component interface
        """
        super().__init__()
        self.objectives = nn.ModuleList(objectives)
        self.constraints = nn.ModuleList(constraints)
        self.components = nn.ModuleList(components)
        self._check_unique_names()

    def _check_unique_names(self):
        num_unique = len(set([o.name for o in self.objectives] + [c.name for c in self.constraints]))
        num_objectives = len(self.objectives) + len(self.constraints)
        assert num_unique == num_objectives, "All objectives and constraints must have unique names."

    def _calculate_loss(self, variables: Dict[str, torch.Tensor]) -> torch.Tensor:
        """

        """
        outputs = {}
        loss = 0.0
        for objective in self.objectives:
            outputs[objective.name] = objective(variables)
            loss += outputs[objective.name]
        for constraint in self.constraints:
            outputs[constraint.name] = constraint(variables)
            loss += outputs[constraint.name]
        return {'loss': loss, **outputs}

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        output_dict = self.step(data)
        output_dict = {**self._calculate_loss(output_dict), **output_dict}
        return {f'{data.name}_{k}': v for k, v in output_dict.items()}

    def step(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for component in self.components:
            output_dict = component(input_dict)
            assert set(output_dict.keys()) - set(input_dict.keys()) == set(output_dict.keys()), \
                f'Name collision in input and output dictionaries, Input_keys: {input_dict.keys()},' \
                f'Output_keys: {output_dict.keys()}'
            input_dict = {**input_dict, **output_dict}
        return input_dict


class MSELoss(Objective):
    def __init__(self, variable_names, weight=1.0, name="ref_loss"):
        super().__init__(
            variable_names,
            nn.functional.mse_loss,
            weight=weight,
            name=name
        )


class RegularizationLoss(Objective):
    def __init__(self, variable_names, weight=1.0, name="reg_loss"):
        super().__init__(
            variable_names,
            lambda *x: torch.sum(*x),
            weight=weight,
            name=name
        )


if __name__ == '__main__':
    nx, ny, nu, nd = 15, 7, 5, 3
    Np = 2
    Nf = 10
    samples = 100
    # Data format: (N,samples,dim)
    x = torch.rand(samples, nx)
    Yp = torch.rand(Np, samples, ny)
    Up = torch.rand(Np, samples, nu)
    Uf = torch.rand(Nf, samples, nu)
    Dp = torch.rand(Np, samples, nd)
    Df = torch.rand(Nf, samples, nd)
    Rf = torch.rand(Nf, samples, ny)
    x0 = torch.rand(samples, nx)


