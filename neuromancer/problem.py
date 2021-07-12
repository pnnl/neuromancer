"""

"""
# python base imports
from typing import Dict, List, Callable

# machine learning/data science imports
import torch
import torch.nn as nn

from neuromancer.constraint import Variable, Objective


class Problem(nn.Module):

    def __init__(self, objectives: List[Objective], constraints: List[Objective],
                 components: List[Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]],
                 variables: List[Variable] = []):
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
        self.variables = variables
        self._check_unique_names()

    def _check_unique_names(self):
        num_unique = len(set([o.name for o in self.objectives] + [c.name for c in self.constraints]))
        num_objectives = len(self.objectives) + len(self.constraints)
        assert num_unique == num_objectives, "All objectives and constraints must have unique names."

    def _calculate_loss(self, variables: Dict[str, torch.Tensor]) -> torch.Tensor:
        """

        """
        loss = 0.0
        for objective in self.objectives:
            variables[objective.name] = objective(variables)
            loss += variables[objective.name]
        for constraint in self.constraints:
            variables[constraint.name] = constraint(variables)
            loss += variables[constraint.name]
        variables['loss'] = loss

    def _evaluate_variables(self, input_dict: Dict[str, torch.Tensor]):
        for variable in self.variables:
            input_dict[variable.name] = variable(input_dict)

    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        output_dict = self.step(data)
        self._calculate_loss(output_dict)
        self._evaluate_variables(output_dict)
        return {f'{data.name}_{k}': v for k, v in output_dict.items()}

    def step(self, input_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for component in self.components:
            output_dict = component(input_dict)
            assert set(output_dict.keys()) - set(input_dict.keys()) == set(output_dict.keys()), \
                f'Name collision in input and output dictionaries, Input_keys: {input_dict.keys()},' \
                f'Output_keys: {output_dict.keys()}'
            input_dict = {**input_dict, **output_dict}
        return input_dict

    def __repr__(self):
        s = "### MODEL SUMMARY ###\n\nCOMPONENTS:"
        if len(self.components) > 0:
            for c in self.components:
                s += f"\n  {repr(c)}"
            s += "\n"
        else:
            s += " none\n"

        s += "\nCONSTRAINTS:"
        if len(self.constraints) > 0:
            for c in self.constraints:
                s += f"\n  {repr(c)}"
            s += "\n"
        else:
            s += " none\n"

        s += "\nOBJECTIVES:"
        if len(self.objectives) > 0:
            for c in self.objectives:
                s += f"\n  {repr(c)}"
            s += "\n"
        else:
            s += " none\n"

        return s


class MSELoss(Objective):
    def __init__(self, variable_names, weight=1.0, name="mse_loss"):
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


