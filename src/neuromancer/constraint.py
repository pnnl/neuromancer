"""
Definition of neuromancer.Constraint class used in conjunction with neuromancer.Variable class. A Constraint has the
same behavior as a Loss but with intuitive syntax for defining via Variable objects.
"""
from typing import Dict, List
import functools
from typing import Callable, Iterable, Union
import copy

import networkx as nx
from plum import dispatch
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from neuromancer.gradients import gradient
from neuromancer.utils import handle_device_placement
import lightning.pytorch as pl




class Loss(nn.Module):
    """
    Drop in replacement for a Constraint object but relies on a list of dictionary keys and a callable function
    to instantiate.
    """
    def __init__(self, input_keys: List[str], loss: Callable[..., torch.Tensor], weight=1.0, name='loss'):
        """

        :param variable_names: List of str
        :param loss: (callable) Number of arguments of the callable should equal the number of strings in variable names.
                                Arguments to callable should be torch.Tensor and return type a 0-dimensional torch.Tensor
        :param weight: (float) Weight of loss for calculating multi-term loss function
        :param name: (str) Name for tracking output
        """
        super().__init__()
        self.name, self.input_keys, self.output_keys = name, input_keys, [name]
        self.weight = weight
        self.loss = loss

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
        return {self.output_keys[0]: self.weight*self.loss(*[variables[k] for k in self.input_keys])}

    def __repr__(self):
        return f"Loss: {self.name}({', '.join(self.input_keys)}) -> {self.loss} * {self.weight}"


class LT(nn.Module):
    """
    Less than constraint for upper bounding the left hand side by the right hand side.
    Used for defining infix operator for the Variable class and calculating constraint
    violation losses for the forward pass of Constraint objects.

    constraint: g(x) <= b
    forward pass returns:
        value = g(x) - b
        penalty = relu(g(x) - b)
        loss = torch.mean(penalty)
    """
    def __init__(self, norm=1):
        super().__init__()
        self.norm = norm

    def __str__(self):
        return 'lt'

    @handle_device_placement
    def forward(self, left, right):
        """

        :param left: torch.Tensor
        :param right: torch.Tensor
        :return: zero dimensional torch.Tensor, torch.Tensor, torch.Tensor
        """

        value = left - right
        penalty = F.relu(value)
        if self.norm == 2:
            penalty = penalty ** 2
        loss = torch.mean(penalty)
        return loss, value, penalty


class GT(nn.Module):
    """
    Greater than constraint for lower bounding the left hand side by the right hand side.
    Used for defining infix operator for the Variable class and calculating constraint
    violation losses for the forward pass of Constraint objects.

    constraint: g(x) >= b
    forward pass returns:
        value = b - g(x)
        penalty = relu(b - g(x))
        loss = torch.mean(penalty)
    """

    def __init__(self, norm=1):
        super().__init__()
        self.norm = norm

    def __str__(self):
        return 'gt'

    @handle_device_placement
    def forward(self, left, right):
        """

        :param left: torch.Tensor
        :param right: torch.Tensor
        :return: zero dimensional torch.Tensor, torch.Tensor, torch.Tensor
        """
        value = right - left
        penalty = F.relu(value)
        if self.norm == 2:
            penalty = penalty ** 2
        loss = torch.mean(penalty)
        return loss, value, penalty


class Eq(nn.Module):
    """
    Equality constraint penalizing difference between left and right hand side.
    Used for defining infix operator for the Variable class and calculating constraint
    violation losses for the forward pass of Constraint objects.

    constraint: g(x) == b
    forward pass returns:
        value = g(x) - b
        penalty = g(x) - b
        loss = torch.mean(penalty)
    """
    def __init__(self, norm=1):
        super().__init__()
        self.norm = norm

    def __str__(self):
        return 'eq'

    @handle_device_placement
    def forward(self, left, right):
        """

        :param left: torch.Tensor
        :param right: torch.Tensor
        :return: zero dimensional torch.Tensor, torch.Tensor, torch.Tensor
        """
        #right = right.type_as(left)
        
        value = left - right
        if self.norm == 1:
            penalty = torch.abs(value)
            loss = F.l1_loss(left, right)
        elif self.norm == 2:
            penalty = value**2
            loss = F.mse_loss(left, right)
        return loss, value, penalty


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
        assert type(var) is Variable, f'{var} must be Variable type'
        if name is None:
            name = f'{var.display_name}_{metric}'
        key = f'{var.key}_{metric}'
        self.input_keys, self.output_keys, self.name = var.keys, [key], name
        self.var = var
        self.metric = metric
        self.weight = weight

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
        return {self.output_keys[0]: self.weight*self.metric(self.var(input_dict))}

    def __repr__(self):
        return f"Objective: {self.name}({', '.join(self.input_keys)}) = {self.weight} * {self.metric}({', '.join(self.input_keys)})"

    def __mul__(self, weight):
        return Objective(self.var, self.metric, self.weight*weight, self.name)

    def __rmul__(self, weight):
        return Objective(self.var, self.metric, self.weight*weight, self.name)


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
            if isinstance(left, (int, float, complex, bool)):
                display_name = str(left)
            else:
                display_name = str(id(left))
            if not isinstance(left, torch.Tensor):
                left = torch.tensor(left)
            left = variable(left, display_name=display_name)
        if not type(right) is Variable:
            if isinstance(right, (int, float, complex, bool)):
                display_name = str(right)
            else:
                display_name = str(id(right))
            if not isinstance(right, torch.Tensor):
                right = torch.tensor(right)
            right = variable(right, display_name=display_name)
        if name is None:
            name = f'{left.display_name} {comparator} {right.display_name}'
        self.key = f'{left.key}_{comparator}_{right.key}'
        input_keys = left.keys + right.keys
        output_keys = [self.key, f'{self.key}_value', f'{self.key}_violation']
        self.input_keys, self.output_keys, self.name = input_keys, output_keys, name
        self.left = left
        self.right = right
        self.comparator = comparator
        self.weight = weight

    def update_name(self, name):
        self.name = name
        self.key = name
        self.output_keys = [name, f'{name}_value', f'{name}_violation']

    @property
    def variable_names(self):
        return [self.left.display_name, self.right.display_name]

    def __xor__(self, norm):
        comparator = type(self.comparator)(norm=norm)
        return Constraint(self.left, self.right, comparator, weight=self.weight, name=self.name)

    def __mul__(self, weight):
        return Constraint(self.left, self.right, self.comparator, weight=self.weight*weight, name=self.name)

    def __rmul__(self, weight):
        return Constraint(self.left, self.right, self.comparator, weight=self.weight*weight, name=self.name)

    def __bool__(self):
        return self.left is self.right

    def grad(self, input_dict, input_key=None):
        """
         returns gradient of the loss w.r.t. input key

        :param input_dict: (dict, {str: torch.Tensor}) Should contain keys corresponding to self.variable_names
        :param input_key: (str) Name of variable in input dict to take gradient with respect to.
        :return: (torch.Tensor)
        """
        return gradient(self.forward(input_dict)[self.key], input_dict[input_key])

    def forward(self, input_dict):
        """

        :param input_dict: (dict, {str: torch.Tensor}) Should contain keys corresponding to self.variable_names
        :return: 0-dimensional torch.Tensor that can be cast as a floating point number
        """
        if isinstance(self.left, Variable):
            left = self.left(input_dict)
            if not isinstance(left, torch.Tensor):
                left = torch.tensor(left)
        if isinstance(self.right, Variable):
            right = self.right(input_dict)
            if not isinstance(right, torch.Tensor):
                right = torch.tensor(right)
        loss, value, violation = self.comparator(left, right)
        output = {name: tensor for tensor, name
                  in zip([self.weight*loss, value, violation], self.output_keys)}
        return output


class Variable(nn.Module):
    """
    Variable is an abstraction that allows for the definition of constraints and objectives with some nice
    syntactic sugar. When a Variable object is called given a dictionary a pytorch tensor is returned, and when
    a Variable object is subjected to a comparison operator a Constraint is returned. Mathematical operators return
    Variables which will instantiate and perform the sequence of mathematical operations. PyTorch callables
    called with variables as inputs return variables.
    Supported infix operators (variable * variable, variable * numeric): +, -, *, @, **, /, <, <=, >, >=, ==, ^
    """

    def __init__(self, input_variables=[], func=None, key=None, display_name=None, value=None):
        """

        :param input_variables: (Variable or torch.Tensor) The Variable arguments to be used in the callable.
        :param func: (Callable) Ideally this callable will take in Tensors and return Tensors
        :param key: (str) Used for retrieving values from a dictionary of {str: Tensor}
                    if key is provided _is_input set to True
        :param display_name: (str) Used only in __repr__ and plotting the computational graph
        :param value: (torch.Tensor, or numpy array, or other python float, int)
                       Value for the node. Can be a trainable parameter
        """
        super().__init__()

        self._func = func
        if isinstance(value, torch.Tensor) and value.requires_grad:
            value = nn.Parameter(value)
        self._value = value
        self._g, self.ordered_nodes = self.make_graph(input_variables)

        self._is_input = key is not None
        self.key = key
        self._display_name = display_name
    
    def make_graph(self, input_variables):
        """
        This is the function that composes the graph of the Variable from constituent input variables which
        are in-nodes to the Variable. It first builds an empty graph then adds itself to the graph.
        Then it goes through the inputs and instantiates Variable objects for them if they are not
        already a Variable. Then it combines the graphs of all Variables by unioning the sets of nodes and edges.
        In the penultimate step edges are added to the graph from the inputs to the Variable being instantiated,
        taking care to shallow copy nodes when there is more than one edge between nodes. Finally, the graph is
        topologically sorted for swift evaluation of the directed acyclic graph.

        :param input_variables: List of arbitrary inputs for self._func
        :return: A topologically sorted list of Variable objects
        """
        g = nx.DiGraph()
        g.add_node(self)

        _input_variables = []
        for i in input_variables:
            if isinstance(i, Variable):
                _input_variables.append(i)
            elif isinstance(i, torch.Tensor):
                _input_variables.append(Variable(value=i, display_name=str(i)))
            else:
                _input_variables.append(Variable(input_variables=[],
                                                 func=functools.partial(lambda x: x, i),
                                                 display_name=str(i)))
        input_variables = _input_variables
        g = nx.compose_all([g] + [i._g for i in input_variables])

        _input_variables = []
        # For operations on variables like (x + x)[1:] need to shallow copy nodes
        # so we can have more that one edge between a node and itself (e.g., one for add and one for slice)
        for i in input_variables:
            if i not in _input_variables:
                _input_variables += [i]
            else:
                _input_variables += [copy.copy(i)]
        edges = [(i, self) for i in _input_variables]
        g.add_edges_from(edges)
        # self Can't be part of ordered nodes since this will make a loop when retrieving parameters
        ordered_nodes = nn.ModuleList(nx.topological_sort(g))[:-1]
        return g, ordered_nodes
   

    
            
    @property
    def display_name(self):
        name = self._display_name
        if self._display_name is None:
            name = self.key
        return name

    @property
    def key(self):
        """
        Used by input Variables to retrieve Tensor values from a dictionary.
        Will be used as a display_name if display_name is not provided to __init__
        :return: (str) String intended to be a key in a dict {str: Tensor}
        """
        return self._key

    @key.setter
    def key(self, k):
        if k is None:
            self._key = str(id(self))
        else:
            self.check_keys(k)
            self._key = k

    def check_keys(self, k):
        assert k not in {n._key for n in self.ordered_nodes}, f'Key {k} repeats existing key. Keys should be unique.'

    @property
    def keys(self):
        keys = [self._key] if self._is_input else []
        return [n._key for n in self.ordered_nodes if n._is_input] + keys

    def __hash__(self):
        """
        This function is needed for pytorch compatibility for some reason.
        """
        return id(self)

    def __add__(self, other):
        return Variable(input_variables=[self, other], func=lambda x, y: x + y, display_name="+")

    def __radd__(self, other):
        return Variable(input_variables=[other, self], func=lambda x, y: x + y, display_name="+")

    def __neg__(self):
        return Variable(input_variables=[self], func=lambda x: -x, display_name="neg")

    def __sub__(self, other):
        return Variable(input_variables=[self, other], func=lambda x, y: x - y, display_name="-")

    def __rsub__(self, other):
        return Variable(input_variables=[other, self], func=lambda x, y: x - y, display_name="-")

    def __mul__(self, other):
        return Variable(input_variables=[self, other], func=lambda x, y: x * y, display_name="∗")

    def __rmul__(self, other):
        return Variable(input_variables=[other, self], func=lambda x, y: x * y, display_name="∗")

    def __matmul__(self, other):
        return Variable(input_variables=[self, other], func=lambda x, y: x @ y, display_name="@")

    def __rmatmul__(self, other):
        return Variable(input_variables=[other, self], func=lambda x, y: x @ y, display_name="@")

    def __truediv__(self, other):
        return Variable(input_variables=[self, other], func=lambda x, y: x / y, display_name="/")

    def __rtruediv__(self, other):
        return Variable(input_variables=[other, self], func=lambda x, y: x / y, display_name="/")

    def __floordiv__(self, other):
        return Variable(input_variables=[self, other], func=lambda x, y: x // y, display_name="//")

    def __rfloordiv__(self, other):
        return Variable(input_variables=[other, self], func=lambda x, y: x // y, display_name="//")

    def __getitem__(self, key):
        return Variable(input_variables=[self], func=lambda x: x[key], display_name="slice")

    def __pow__(self, other):
        return Variable(input_variables=[self, other], func=lambda x, y: x**y, display_name="pow")

    def __rpow__(self, other):
        return Variable(input_variables=[other, self], func=lambda x, y: x**y, display_name="pow")

    def __abs__(self):
        return Variable(input_variables=[self], func=lambda x: abs(x), display_name="abs")

    def __mod__(self, modulo):
        return Variable(input_variables=[self, modulo], func=lambda x, y: x % y, display_name="mod")

    def __rmod__(self, modulo):
        return Variable(input_variables=[modulo, self], func=lambda x, y: x % y, display_name="mod")

    def show(self, figname=None):
        """
        Plot and save computational graph

        :param figname: (str) Name to save figure to.
        """
        nx.draw(self._g, with_labels=True)
        if figname is not None:
            plt.savefig(figname)
            plt.close()
        else:
            plt.show()


    @property
    def T(self):
        return Variable(input_variables=[self], func=lambda x: x.T, display_name="T")

    @property
    def mT(self):
        return Variable(input_variables=[self], func=lambda x: x.mT, display_name="mT")

    def __eq__(self, other):
        return Constraint(self, other, Eq())

    def __lt__(self, other):
        return Constraint(self, other, LT())

    def __le__(self, other):
        return Constraint(self, other, LT())

    def __gt__(self, other):
        return Constraint(self, other, GT())

    def __ge__(self, other):
        return Constraint(self, other, GT())

    def __repr__(self) -> str:
        return self.display_name

    def forward(self, datadict=None):
        """
        Forward pass goes through topologically sorted nodes calculating or retrieving values.

        :param datadict: (dict, {str: Tensor}) Optional dictionary for Variable graphs which take input
        :return: (torch.Tensor) Tensor value from evaluating the variable's computational graph.
        """
        datadict = {} if datadict is None else datadict
        for n in self.ordered_nodes:
            self.get_value(n, datadict)
        self.get_value(self, datadict)
        return self._value

    def get_value(self, n, datadict):
        if not n._is_input:
            if n._func is not None:
                args = [src._value for src, _ in self._g.in_edges(n)]
                n._value = n._func(*args)
        else:
            n._value = datadict[n._key]
        datadict[n.key] = n._value

    @dispatch
    def unpack(self, nret: int):
        """
        Creates new variables for a node that evaluates to multiple values.
        This is useful for unpacking results of functions that return multiple values such as `torch.linalg.svd`:

        :param nret: (int) Number of return values from the torch function
        :return: [Variable] List of Variable objects for each value returned by the torch function
        """

        return [variable([self], functools.partial(lambda i, x: x[i], idx))
                for idx in range(nret)]

    @dispatch
    def unpack(self, names: Iterable[str]):
        """
        Creates new variables for a node that evaluates to multiple values.
        This is useful for unpacking results of functions that return multiple values such as `torch.linalg.svd`:

        ```
        m = Variable("m", torch.ones(10,10))
        u, s, v = torch.linalg.svd(m).unpack(["u","s","v"])
        ```
        """
        return [variable([self], functools.partial(lambda i, x: x[i], idx), display_name=k)
                for idx, k in enumerate(names)]

    # Compatabiliy with PyTorch
    # https://pytorch.org/docs/stable/notes/extending.html#extending-torch
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs

        variables = tuple([a for a in args if isinstance(a, Variable)])
        non_variable_args = tuple([a for a in args if not isinstance(a, Variable)])

        def wrapped(*call_args, **call_kwargs):
            all_args = non_variable_args + call_args
            all_kwargs = kwargs | call_kwargs
            return func(*all_args, **all_kwargs)

        return variable(variables, wrapped, display_name=func.__name__)

    def grad(self, other):
        return variable([self, other], gradient, display_name=f'd{self.display_name}/d{other.display_name}')

    @property
    def value(self):
        return self._value

    def minimize(self, metric=torch.mean, weight=1.0, name=None):
        return Objective(self, metric=metric, weight=weight, name=name)


_size = Union[torch.Size, Iterable[int]]
_name = Union[str, None]
_input = Union[Variable, float, int, torch.Tensor]


@dispatch
def variable(display_name=None) -> Variable:  # pylint: disable=function-redefined
    """
    For instantiating a trainable Variable. returns Variable with trainable value = 0dim Tensor from std. normal dist.

    :param display_name: (str) for plotting graph and __repr__
    :return: Variable with value = 0 dimensional nn.Parameter with requires_grad=True
    """
    t = torch.randn(1, requires_grad=True)
    return Variable(display_name=display_name, value=t)


@dispatch
def variable(key: _name) -> Variable:
    """
    Canonical way to instantiate an input Variable

    :param key: (str) key for indexing value out of dictionary
    :return: input Variable
    """
    def raise_err():
        raise RuntimeError("eval_node should never be called on an input_node")
    func = raise_err
    return Variable(key=key, func=func)


@dispatch
def variable(*size: int, display_name: _name = None) -> Variable:  # pylint: disable=function-redefined
    """

    :param size: Sequence of integer arguments describing shape of parameter
    :param display_name: (str) for plotting graph and __repr__
    :return: Variable with value = nn.Parameter with shape=[size], with requires_grad=True
    """
    t = torch.randn(size, requires_grad=True)
    return Variable(display_name=display_name, value=t)


@dispatch
def variable(size: _size, key: _name = None, display_name=None) -> Variable:  # pylint: disable=function-redefined

    """

    :param size: Iterable of integer arguments describing shape of parameter
    :param display_name: (str) for plotting graph and __repr__
    :return: Variable with value = nn.Parameter with shape=size, with requires_grad=True
    """

    t = torch.randn(size, requires_grad=True)
    return Variable(display_name=display_name, value=t)


@dispatch
def variable(value: torch.Tensor, display_name=None) -> Variable:  # pylint: disable=function-redefined
    """

    :param value: (Tensor) Value to be retrieved when called. Can be a trainable parameter.
    :param display_name: (str) for plotting graph and __repr__
    :return: Variable with value = value. Value will be wrapped with nn.Parameter if requires_grad=True
    """
    return Variable(display_name=display_name, value=value)


@dispatch
def variable(inputs: Iterable[_input], func: Callable, display_name=None) -> Variable:  # pylint: disable=function-redefined
    """
    Create a variable with arbitrary function and arbitrary inputs

    :param inputs: (Iterable which can contain mix of integer, float, torch.Tensor, and Variable objects) Input to the function.
    :param func: A Callable which returns torch.Tensor objects
    :param display_name: (str) for plotting graph and __repr__
    :return: Variable which will evaluate computational graph when called with dictionary containing input key:value pairs
    """
    return Variable(input_variables=inputs, func=func, display_name=display_name)