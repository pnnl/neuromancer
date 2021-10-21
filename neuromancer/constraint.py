"""
Definition of neuromancer.Constraint class used in conjunction with neuromancer.Variable class. A Constraint has the
same behavior as a Loss but with intuitive syntax for defining via Variable objects.
"""
from typing import Dict, List, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuromancer.gradients import gradient


class Loss(nn.Module):
    """
    Drop in replacement for a Constraint object but relies on a list of dictionary keys and a callable function
    to instantiate.
    """
    def __init__(self, variable_names: List[str], loss: Callable[..., torch.Tensor], weight=1.0, name='loss'):
        """

        :param variable_names: List of str
        :param loss: (callable) Number of arguments of the callable should equal the number of strings in variable names.
                                Arguments to callable should be torch.Tensor and return type a 0-dimensional torch.Tensor
        :param weight: (float) Weight of loss for calculating multi-term loss function
        :param name: (str) Name for tracking output
        """
        super().__init__()
        self.variable_names = variable_names
        self.weight = weight
        self.loss = loss
        self.name = name

    def grad(self, variables, input_key=None):
        """
         returns gradient of the loss w.r.t. input variables

        :param variables:
        :param input_key: string
        :return:
        """
        return gradient(self.forward(variables)[self.name], variables[input_key])

    def forward(self, variables: Dict[str, torch.Tensor]) -> torch.Tensor:
        """

        :param variables: (dict, {str: torch.Tensor}) Should contain keys corresponding to self.variable_names
        :return: 0-dimensional torch.Tensor that can be cast as a floating point number
        """
        return {self.name: self.weight*self.loss(*[variables[k] for k in self.variable_names])}

    def __repr__(self):
        return f"Loss: {self.name}({', '.join(self.variable_names)}) -> {self.loss} * {self.weight}"


class LT(nn.Module):
    """
    Less than constraint for upper bounding the left hand side by the right hand side.
    Used for defining infix operator for the Variable class and calculating constraint
    violation losses for the forward pass of Constraint objects.
    """
    def __init__(self, norm=1):
        super().__init__()
        self.norm = norm

    def __str__(self):
        return 'lt'

    def forward(self, left, right):
        """

        :param left: torch.Tensor
        :param right: torch.Tensor
        :return: zero dimensional torch.Tensor
        """
        if self.norm == 1:
            return torch.mean(F.relu(left - right))
        elif self.norm == 2:
            return torch.mean((F.relu(left - right))**2)


class GT(nn.Module):
    """
    Greater than constraint for lower bounding the left hand side by the right hand side.
    Used for defining infix operator for the Variable class and calculating constraint
    violation losses for the forward pass of Constraint objects.
    """

    def __init__(self, norm=1):
        super().__init__()
        self.norm = norm

    def __str__(self):
        return 'gt'

    def forward(self, left, right):
        """

        :param left: torch.Tensor
        :param right: torch.Tensor
        :return: zero dimensional torch.Tensor
        """
        if self.norm == 1:
            return torch.mean(F.relu(right - left))
        elif self.norm == 2:
            return torch.mean((F.relu(right - left)) ** 2)


class Eq(nn.Module):
    """
    Equality constraint penalizing difference between left and right hand side.
    Used for defining infix operator for the Variable class and calculating constraint
    violation losses for the forward pass of Constraint objects.
    """
    def __init__(self, norm=1):
        super().__init__()
        self.norm = norm

    def __str__(self):
        return 'eq'

    def forward(self, left, right):
        """

        :param left: torch.Tensor
        :param right: torch.Tensor
        :return: zero dimensional torch.Tensor
        """
        if self.norm == 1:
            return F.l1_loss(left, right)
        elif self.norm == 2:
            return F.mse_loss(left, right)


class Objective(nn.Module):
    """
    Drop in replacement for a Loss object constructed via neuromancer Variable object
    in the forward pass evaluates metric as torch function on Variable values
    """
    def __init__(self, var, metric=torch.mean, weight=1.0, name=None):
        """

        :param var: (nm.Variable) expression to be minimized
        :param metric: (torch function) differentiable scalar valued function to penalize the expression
        :param weight: (float, int, or zero-D torch.Tensor) For scaling calculated Constraint violation loss
        :param name: (str) Optional intuitive name for storing in Problem's output dictionary.
        """
        super().__init__()
        if not type(var) is Variable:
            var = Variable(str(var), value=var)
        self.var = var
        self.metric = metric
        self.weight = weight
        if name is None:
            self.name = f'{self.var.name}_{self.metric}'
        else:
            self.name = name

    @property
    def variable_names(self):
        return [self.var.name]

    def grad(self, input_dict, input_key=None):
        """
         returns gradient of the loss w.r.t. input variables

        :param input_dict:
        :param input_key: string
        :return:
        """
        return gradient(self.forward(input_dict)[self.name], input_dict[input_key])

    def forward(self, input_dict):
        """

        :param input_dict: (dict, {str: torch.Tensor}) Should contain keys corresponding to self.variable_names
        :return:  (dict, {str: 0-dimensional torch.Tensor}) tensor value can be cast as a floating point number
        """
        return {self.name: self.weight*self.metric(self.var(input_dict))}

    def __repr__(self):
        return f"Objective: {self.name}({', '.join(self.variable_names)}) = {self.weight} * {self.metric}({', '.join(self.variable_names)})"


class Constraint(nn.Module):
    """
    Drop in replacement for a Loss object but constructed by a composition of Variable objects
    using comparative infix operators, '<', '>', '==', '<=', '>=' and '*' to weight loss component and '^' to
    determine l-norm of constraint violation in determining loss.
    """
    def __init__(self, left, right, comparator, weight=1.0, name=None):
        """

        :param left: (nm.Variable or numeric) Left hand side of equality or inequality constraint
        :param right: (nm.Variable or numeric) Right hand side of equality or inequality constraint
        :param comparator: (nn.Module) Intended to be LE, GE, LT, GT, or Eq object, but can be any nn.Module
                                       which satisfies the Comparator interface (init function takes an integer norm and
                                       object has an integer valued self.norm attribute.
        :param weight: (float, int, or zero-D torch.Tensor) For scaling calculated Constraint violation loss
        :param name: (str) Optional intuitive name for storing in Problem's output dictionary.
        """
        super().__init__()
        if not type(left) is Variable:
            left = Variable(str(left), value=left)
        if not type(right) is Variable:
            right = Variable(str(right), value=right)
        self.left = left
        self.right = right
        self.comparator = comparator
        self.weight = weight
        if name is None:
            self.name = f'{self.left}_{self.comparator}_{self.right}'
        else:
            self.name = name

    @property
    def variable_names(self):
        return [self.left.name, self.right.name]

    def __xor__(self, norm):
        comparator = type(self.comparator)(norm=norm)
        return Constraint(self.left, self.right, comparator, weight=self.weight, name=self.name)

    def __mul__(self, weight):
        return Constraint(self.left, self.right, self.comparator, weight=weight, name=self.name)

    def __rmul__(self, weight):
        return Constraint(self.left, self.right, self.comparator, weight=weight, name=self.name)

    def grad(self, input_dict, input_key=None):
        """
         returns gradient of the loss w.r.t. input key

        :param input_dict: (dict, {str: torch.Tensor}) Should contain keys corresponding to self.variable_names
        :param input_key: (str) Name of variable in input dict to take gradient with respect to.
        :return: (torch.Tensor)
        """
        return gradient(self.forward(input_dict)[self.name], input_dict[input_key])

    def forward(self, input_dict):
        """

        :param input_dict: (dict, {str: torch.Tensor}) Should contain keys corresponding to self.variable_names
        :return: 0-dimensional torch.Tensor that can be cast as a floating point number
        """
        return {self.name: self.weight*self.comparator(self.left(input_dict), self.right(input_dict))}


class Variable(nn.Module):
    """
    Variable is an abstraction that allows for the definition of constraints and objectives with some nice
    syntactic sugar. When a Variable object is called given a dictionary a pytorch tensor is returned, and when
    a Variable object is subjected to a comparison operator a Constraint is returned. Mathematical operators return
    Variables which will instantiate and perform the sequence of mathematical operations.

    Supported infix operators (variable * variable, variable * numeric): +, -, *, @, **, <, <=, >, >=, ==, ^
    """

    def __init__(self, key, value=None, left=None, right=None, operator=None, slice=None, name=None):
        """

        :param key: (str) Unique string identifier which should occur as a key at some point in
                          the data dictionary which is accumulated through the computational graph unless
                          the variable is a composition of other variables with a numeric operator.
        :param value: (torch.Tensor or numeric) To use when variable is a constant or nn.Parameter
        :param left: (Variable) Another Variable used in the composition of this Variable
        :param right: (Variable) Another Variable used in the composition of this Variable
        :param operator: (str) Indicates how this Variable is derived from left and right
        :param slice: (Slice) Indicates if and where Variable is to be indexed
        :param name: (str) Optional intuitive name for Variable for storage in a Problem's output dictionary.
        """
        super().__init__()

        self.key = key
        self.left = left
        self.right = right
        self.op = operator
        if value is not None and not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32)
        self.value = value
        self.slice = slice
        self._check_()
        if name is None:
            if slice is None:
                self.name = self.key
            else:
                self.name = f'{self.key}_{str(slice)}'
        else:
            self.name = name

    def _check_(self):
        """
        Some things should never happen.
        """
        if self.left is not None:
            assert type(self.left) is Variable
        if self.right is not None:
            assert type(self.right) is Variable
        if self.left is not None or self.right is not None:
            assert self.op is not None

    def forward(self, data):
        """
        The call function is going to hand back a pytorch tensor. In the base case the call function will simply look
        up the tensor in the data dictionary. More complicated cases will involve calling the left and calling the right
        (recursively traversing the tree defining this composition ov Variables) and then performing a pytorch operation.

        :param data: (dict: {str: torch.Tensor})
        :return: torch.Tensor
        """
        data_tensors = [v for v in data.values() if isinstance(v, torch.Tensor)]
        if self.value is not None:
            if data_tensors:
                self.value.to(data_tensors[0].device)
            value = self.value
        elif self.op == 'add':
            value = self.left(data) + self.right(data)
        elif self.op == 'sub':
            value = self.left(data) - self.right(data)
        elif self.op == 'mul':
            value = self.left(data) * self.right(data)
        elif self.op == 'pow':
            value = self.left(data) ** self.right(data)
        elif self.op == 'matmul':
            value = self.left(data) @ self.right(data)
        elif self.op == 'neg':
            value = -self.left(data)
        elif self.op == 'div':
            value = self.left(data) / self.right(data)
        elif self.op == 'grad':
            value = gradient(self.left(data), self.right(data))
        else:
            if self.slice is not None:
                key = self.key[:-len(str(self.slice))-1]
                value = data[key]
            else:
                key = self.key
                return data[key]

        if self.slice is None:
            data[self.key] = value
            return data[self.key]
        else:
            data[self.key] = value[self.slice]
            return data[self.key]

    def __getitem__(self, slice):

        return Variable(f'{self.key}_{str(slice)}', value=self.value, left=self.left,
                        right=self.right, operator=self.op, slice=slice)

    def __str__(self):
        if self.value is not None:
            return f'{self.key}={self.value}'
        else:
            return self.key

    def __repr__(self):
        return f'neuromancer.variable.Variable object, key={self.key}, left={self.left}, ' \
               f'right={self.right}, operator={self.op}, value={self.value}, slice={self.slice}'

    """
    Numeric operators. Numeric operators return a variable with a special __call__ function which first retrieves instantiated
    numeric quantities for the left and right hand side and performs the corresponding pytorch operator on the instantiate quantities
    """
    def __add__(self, other):
        if not type(other) is Variable:
            other = Variable(str(other), value=other)
        return Variable(f'{self.key}_plus_{other.key}', left=self, right=other, operator='add')

    def __radd__(self, other):
        if not type(other) is Variable:
            other = Variable(str(other), value=other)
        return Variable(f'{self.key}_plus_{other.key}', left=other, right=self, operator='add')

    def __sub__(self, other):
        if not type(other) is Variable:
            other = Variable(str(other), value=other)
        return Variable(f'{self.key}_minus_{other.key}', left=self, right=other, operator='sub')

    def __rsub__(self, other):
        if not type(other) is Variable:
            other = Variable(str(other), value=other)
        return Variable(f'{self.key}_minus_{other.key}', left=other, right=self, operator='sub')

    def __mul__(self, other):
        if not type(other) is Variable:
            other = Variable(str(other), value=other)
        return Variable(f'{self.key}_times_{other.key}', left=self, right=other, operator='mul')

    def __rmul__(self, other):
        if not type(other) is Variable:
            other = Variable(str(other), value=other)
        return Variable(f'{self.key}_times_{other.key}', left=other, right=self, operator='mul')

    def __matmul__(self, other):
        if not type(other) is Variable:
            other = Variable(str(other), value=other)
        return Variable(f'{self.key}_matmul_{other.key}', left=self, right=other, operator='matmul')

    def __rmatmul__(self, other):
        if not type(other) is Variable:
            other = Variable(str(other), value=other)
        return Variable(f'{self.key}_matmul_{other.key}', left=other, right=self, operator='matmul')

    def __pow__(self, other):
        if not type(other) is Variable:
            other = Variable(str(other), value=other)
        return Variable(f'{self.key}_pow_{other.key}', left=self, right=other, operator='pow')

    def __rpow__(self, other):
        if not type(other) is Variable:
            other = Variable(str(other), value=other)
        return Variable(f'{self.key}_pow_{other.key}', left=other, right=self, operator='pow')

    def __truediv__(self, other):
        if not type(other) is Variable:
            other = Variable(str(other), value=other)
        return Variable(f'{self.key}_div_{other.key}', left=self, right=other, operator='div')

    def __rtruediv__(self, other):
        if not type(other) is Variable:
            other = Variable(str(other), value=other)
        return Variable(f'{self.key}_div_{other.key}', left=other, right=self, operator='div')

    def __neg__(self):
        return Variable(f'neg_{self.key}', left=self, operator='neg')

    def grad(self, other):
        return Variable(f'd{self.key}/d{other.key}', left=self, right=other, operator='grad')

    """
    Comparison operators. When a variable and numeric value (float, int, Tensor) are compared a constraint is implicitly 
    defined and a Constraint object is returned. For now <, >, <=, and >= will create an inequality Constraint with 
    an L1 norm penalty on constraint violations. A Constraint object can be made to use an L2 norm penalty with the ^ operator, 
    e.g. (x < 1)^2. x == y will return an equality Constraint equivalent to the two inequality constraints x - y <=0, y-x <=0. 


    != is not a defined comparison for variables  
    """

    def __lt__(self, other):
        return Constraint(self, other, LT())

    def __le__(self, other):
        return Constraint(self, other, LT())

    def __gt__(self, other):
        return Constraint(self, other, GT())

    def __ge__(self, other):
        return Constraint(self, other, GT())

    def __eq__(self, other):
        return Constraint(self, other, Eq())

    def minimize(self, metric=torch.mean, weight=1.0, name=None):
        return Objective(self, metric=metric, weight=weight, name=name)

    def __hash__(self):
        """
        without this hack we have: TypeError: unhashable type: 'Variable'
        https://github.com/pytorch/pytorch/issues/16756
        """
        return nn.Module.__hash__(self)

