"""
Definition of the neuromancer.variable class. Variables are objects associated
with some stage of computation in the computational
graph that the use can define different constraints and objectives on.
"""
import neuromancer.constraint as constraint


class Variable:
    """
    Variable is an abstraction that allows for the definition of constraints and objectives with some nice
    syntactic sugar. When a Variable object is called given a dictionary a pytorch tensor is returned, and when
    a Variable object is subjected to a comparison operator a Constraint is returned. Mathematical operators return
    Variables which will instantiate and perform the sequence of mathematical operations.

    Supported infix operators (variable * variable, variable * numeric): +, -, *, @, **, <, <=, >, >=, ==, ^
    """
    def __init__(self, key, constant=None, left=None, right=None, operator=None):
        """

        :param key: (str) Unique string identifier which should occur as a key at some point in
                          the data dictionary which is accumulated through the computational graph.
        """
        self.key = key
        self.left = left
        self.right = right
        self.op = operator
        self.constant = constant
        self._check_()

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
        if self.op == 'neg':
            assert self.left is None and self.right is None
        if self.left is not None:
            assert self.right is not None
        if self.right is not None:
            assert self.left is not None

    def __call__(self, data):
        """
        The call function is going to hand back a pytorch tensor. In the base case the call function will simply look
        up the tensor in the data dictionary. More complicated cases will involve calling the left and calling the right
        and then performing a pytorch operation.
        :param data: (dict: {str: torch.Tensor})
        :return: torch.Tensor
        """

        if self.constant is not None:
            self.data[self.key] = self.constant
            return self.constant
        elif self.op == 'add':
            return self.left(data) + self.right(data)
        elif self.op == 'sub':
            return self.left(data) - self.right(data)
        elif self.op == 'pow':
            return self.left(data)**self.right(data)
        elif self.op == 'matmul':
            return self.left(data) @ self.right(data)
        elif self.op == 'neg':
            return -data[self.key.strip('neg_')]
        else:
            return data[self.key]

    """
        Numeric operators. Numeric operators return a variable with a special __call__ function which first retrieves instantiated
        numeric quantities for the left and right hand side and performs the corresponding pytorch operator on the instantiate quantities
        """

    def __str__(self):
        if self.constant is not None:
            return f'{self.key}={self.constant}'
        else:
            return self.key

    def __repr__(self):
        return f'neuromancer.variable.Variable object, key={self.key}, left={self.left}, right={self.right}, operator={self.operator}, constant={self.constant}'

    def __add__(self, other):
        if not type(other) is Variable:
            other = Variable(str(other), constant=other)
        return Variable(f'{self.key}_plus_{other.key}', left=self, right=other, operator='add')

    def __radd__(self, other):
        if not type(other) is Variable:
            other = Variable(str(other), constant=other)
        return Variable(f'{self.key}_plus_{other.key}', left=self, right=other, operator='add')

    def __sub__(self, other):
        if not type(other) is Variable:
            other = Variable(str(other), constant=other)
        return Variable(f'{self.key}_minus_{other.key}', left=self, right=other, operator='sub')

    def __rsub__(self, other):
        if not type(other) is Variable:
            other = Variable(str(other), constant=other)
        return Variable(f'{self.key}_minus_{other.key}', left=self, right=other, operator='sub')

    def __mul__(self, other):
        if not type(other) is Variable:
            other = Variable(str(other), constant=other)
        return Variable(f'{self.key}_times_{other.key}', left=self, right=other, operator='mul')

    def __rmul__(self, other):
        if not type(other) is Variable:
            other = Variable(str(other), constant=other)
        return Variable(f'{self.key}_times_{other.key}', left=self, right=other, operator='mul')

    def __matmul__(self, other):
        if not type(other) is Variable:
            other = Variable(str(other), constant=other)
        return Variable(f'{self.key}_matmul_{other.key}', left=self, right=other, operator='matmul')

    def __rmatmul__(self, other):
        if not type(other) is Variable:
            other = Variable(str(other), constant=other)
        return Variable(f'{self.key}_matmul_{other.key}', left=self, right=other, operator='matmul')

    def __pow__(self, other):
        if not type(other) is Variable:
            other = Variable(str(other), constant=other)
        return Variable(f'{self.key}_pow_{other.key}', left=self, right=other, operator='pow')

    def __rpow__(self, other):
        if not type(other) is Variable:
            other = Variable(str(other), constant=other)
        return Variable(f'{self.key}_pow_{other.key}', left=self, right=other, operator='pow')

    def __neg__(self):
        return Variable(f'neg_{self.key}', operator='neg')

    """
    Comparison operators. When a variable and numeric value (float, int, Tensor) are compared a constraint is implicitly 
    defined and a Constraint object is returned. For now <, >, <=, and >= will create an inequality Constraint with 
    an L1 norm penalty on constraint violations. A Constraint object can be made to use an L2 norm penalty with the ^ operator, 
    e.g. (x < 1)^2. x == y will return an equality Constraint equivalent to the two inequality constraints x - y <=0, y-x <=0. 
    
    
    != is not a defined comparison for variables  
    """
    def __lt__(self, other):
        return constraint.Constraint(self, other, constraint.LT())

    def __le__(self, other):
        return constraint.Constraint(self, other, constraint.LT())

    def __gt__(self, other):
        return constraint.Constraint(self, other, constraint.GT())

    def __ge__(self, other):
        return constraint.Constraint(self, other, constraint.GT())

    def __eq__(self, other):
        return constraint.Constraint(self, other, constraint.Eq())


if __name__ == '__main__':
    x = Variable('x')
    y = Variable('y')
    print(x + y)
    print(x - y)
    print(x * y)
    print(x + 1)
    print(1 + x)
    print(1 + x < 0)





