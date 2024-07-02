import numpy as np
import itertools
import torch

class FunctionLibrary:

    def __init__(
        self,
        functions,
        n_features,
        function_names=None
    ):
        """
        :param functions: (list) a one-dimensional list of callable functions.
                                 All callables must be able to operate on the
                                 columns of a two-dimensional torch.Tensor of
                                 floating point numbers.
        :param n_features: (int) number of columns of data matrix
        :param function_names: (list) name for each function, must either be None
                                      or a list of the same length as the functions
        """
        if function_names is not None:
            assert len(functions) == len(function_names), "Must have one name for each function"

        assert isinstance(functions, list), "Functions must be passed as list"

        self.library = functions

        self.shape = (len(self.library), n_features)
        self.function_names = function_names

    def evaluate(self, x):
        """
        :param x: (torch.Tensor) an array of values to evaluate the functions at
        :return: (torch.Tensor, shape=[[# of rows of X, number of functions) the 
                       functions evaluated at every single time step of the data
        """
        output = torch.zeros(size=(x.shape[0], self.shape[0]))

        for i in range(self.shape[0]):
            output[:, i] = self.library[i](x)

        return output

    def __str__(self):
        """
        :return: (str) all of the names provided to the library, or generic function names
        """
        out_str = ""
        if self.function_names is not None:
            for name in self.function_names:
                out_str += f"{name}, "

        else:
            for i in range(self.shape[0]):
                out_str += f"f{i}, "
        return out_str[:-2]


class PolynomialLibrary(FunctionLibrary):
    def __init__(
        self,
        n_features,
        max_degree=2
        ):
        """
        :param n_features: (int) the number of features (columns) in the input dataset
        :param max_degree: (int) the maximum total degree of any polynomial 
                                 (i.e. max_degree=3 -> x^2y)
        """
        self.max_degree = max_degree
        lib, function_names = self.__create_library(n_features)
        super().__init__(lib, n_features, function_names)

    def __create_library(self, n_features):
        """
        :param n_features: (int) the number of features (columns) in the input dataset
        :return: (list) a row vector of all of the polynomial functions
        :return: (list) a list of all of the names of the functions
        """
        funs_list = [(lambda y: lambda X: X[:, y])(i) for i in range(n_features)]
        vars_list = [f"x{i}" for i in range(n_features)]
        all_combos = [(lambda X: torch.ones(size=(X.shape[0],)),)]
        all_names = []

        for i in range(1, self.max_degree+1):
            combos = list(itertools.combinations_with_replacement(funs_list, i))
            names = list(itertools.combinations_with_replacement(vars_list, i))

            for j in range(len(combos)):
                all_combos.append(combos[j])

            for j in range(len(names)):
                all_names.append(names[j])

        library = all_combos
        names = self.__convert(all_names)
        return library, names

    def __convert(self, names):
        """
        :param names: (list) a list of tuples of all of the names of terms that were combined
        :return: (list) the names converted into a more intrepreted form (i.e. x*x*x -> x^3)
        """
        return_names = ["1"]
        for func in names:
            name = ""
            for term in func:
                if term not in name:
                    name += f"{term}*"
                else:
                    if f"{term}^" in name:
                        degree = int(name[name.find(f"{term}^") + 3])
                        name = name.replace(f"{term}^{degree}", f"{term}^{degree+1}")
                    else:
                        name = name.replace(term, f"{term}^2")
            return_names.append(name[:-1])
        return return_names

    def evaluate(self, X):
        """
        :param X: (torch.Tensor) the two-dimensional dataset to put through the library
        :return: (torch.Tensor, [# of rows of X, number of functions]) the library evaluated at X
        """
        output = torch.ones((X.shape[0], self.shape[0]))
        for i in range(self.shape[0]):
            for func in self.library[i]:
                output[:, i] *= func(X)

        return output


class FourierLibrary(FunctionLibrary):
    def __init__(
        self,
        n_features,
        max_freq=2,
        include_sin=True,
        include_cos=True
    ):
        """
        :param n_features: (int) the number of features of the dataset (columns of data matrix)
        :param max_freq: (int) the max multiplier for the frequency
        :param include_sin: (bool) include the sine function
        :param include_cos: (bool) include the cosine function
        """
        self.max_freq = max_freq
        self.include_sin = include_sin
        self.include_cos = include_cos
        lib, function_names = self.__create_library(n_features)
        super().__init__(lib, n_features, function_names)

    def __create_library(self, n_features):
        """
        :param n_features: (int) the number of input features of the dataset (columns of data matrix)
        :return: (list) an array of functions of sines and cosines
        :return: (list) a list of the names of each function (i.e. sin(2*x0))
        """
        sines = []
        sin_names = []
        cosines = []
        cos_names = []
        if self.include_sin:
            sines = [(lambda z: [(lambda y: lambda X: torch.sin(y * X[:,z]))(i) \
                    for i in range(1,self.max_freq+1)]) (j) for j in \
                    range(n_features)]
            sin_names = [[f"sin({j}*x{i})"  for j in \
                    range(1, self.max_freq+1)] for i in range(n_features)]

        if self.include_cos:
            cosines = [(lambda z: [(lambda y: lambda X: torch.cos(y * X[:,z]))(i)\
                    for i in range(1,self.max_freq+1)]) (j) for j in \
                    range(n_features)]
            cos_names = [[f"cos({j}*x{i})"  for j in \
                    range(1, self.max_freq+1)] for i in range(n_features)]

        library = sum(sines, []) + sum(cosines, [])
        function_names = sum(sin_names, []) + sum(cos_names, [])

        return library, function_names
