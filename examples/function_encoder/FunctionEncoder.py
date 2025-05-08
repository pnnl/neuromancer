from typing import Union, Tuple, List, Callable
import torch

# NOTE: This is a stripped down version of the function encoder code here:
# https://github.com/tyler-ingebrand/FunctionEncoder.git
# See this repo for standalone function encoder code and examples.
class FunctionEncoder(torch.nn.Module):
    """A function encoder learns basis functions/vectors over a Hilbert space.

    A function encoder learns basis functions/vectors over a Hilbert space.
    Typically, this is a function space mapping to Euclidean vectors, but it can be any Hilbert space, IE probability distributions.
    This class has a general purpose algorithm which supports both deterministic and stochastic data.
    The only difference between them is the dataset used and the inner product definition.
    This class supports two methods for computing the coefficients of the basis function, also called a representation:
    1. "inner_product": It computes the inner product of the basis functions with the data via a Monte Carlo approximation.
    2. "least_squares": This method computes the least squares solution in terms of vector operations. This typically trains faster and better.
    This class also supports the residuals method, which learns the average function in the dataset. The residuals/error of this approximation,
    for each function in the space, is learned via a function encoder. This tends to improve performance when the average function is not f(x) = 0.
    """

    def __init__(self,
                 basis_functions: Union[List[torch.nn.Module], torch.nn.Module],
                 average_function: Union[torch.nn.Module, Callable, None] = None,
                 use_least_squares: bool = True
                 ):
        """ Initializes a function encoder.

        Args:
        basis_functions: List[torch.nn.Module]: The basis functions. This is a list of torch modules, typically MLPs.
        average_function: Union[torch.nn.Module, Callable, None]: The average function if using the residuals method.
            This can be a neural network, in which case its learned, or a callback function.
            If None, no average function is used.
        """
        super(FunctionEncoder, self).__init__()

        if type(basis_functions) is torch.nn.Module:
            self.basis_functions = basis_functions
        else:
            self.basis_functions = torch.nn.ModuleList(basis_functions)
        self.average_function = average_function
        self.use_least_squares = use_least_squares

    def compute_representation(self,
                               example_xs: torch.tensor,
                               example_ys: torch.tensor,
                               **kwargs) -> Tuple[torch.tensor, Union[torch.tensor, None]]:
        """Computes the coefficients of the basis functions.

        This method does the forward pass of the basis functions (and the average function if it exists) over the example data.
        Then it computes the coefficients of the basis functions via a Monte Carlo integration of the inner product with the example data.

        Args:
        example_xs: torch.tensor: The input data. Shape (n_example_datapoints, input_size) or (n_functions, n_example_datapoints, input_size)
        example_ys: torch.tensor: The output data. Shape (n_example_datapoints, output_size) or (n_functions, n_example_datapoints, output_size)
        method: str: "inner_product" or "least_squares". Determines how to compute the coefficients of the basis functions.
        kwargs: dict: Additional kwargs to pass to the least squares method.

        Returns:
        torch.tensor: The coefficients of the basis functions. Shape (n_functions, n_basis) or (n_basis,) if n_functions=1.
        Union[torch.tensor, None]: The gram matrix if using least squares method. None otherwise.
        """

        # if not in terms of functions, add a function batch dimension
        reshaped = False
        if len(example_xs.shape) == 2:
            reshaped = True
            example_xs = example_xs.unsqueeze(0)
            example_ys = example_ys.unsqueeze(0)

        # optionally subtract average function if we are using residuals method
        # we dont want to backprop to the average function. So we block grads.
        if self.average_function is not None:
            with torch.no_grad():
                raise Exception("TODO: How do we want to train the average function? \
                we can either train this by simply subtracting the average function, like so, then backproping \
                In the past, I trained the average function independently using a separate supervised loss.  \
                ")
                example_y_hat_average = self.forward_average_function(example_xs)
                example_ys = example_ys - example_y_hat_average

        # compute representation
        Gs = self.forward_basis_functions(example_xs)  # forward pass of the basis functions
        if not self.use_least_squares:
            representation = self._compute_inner_product_representation(Gs, example_ys)
            gram = None
        else:
            representation, gram = self._compute_least_squares_representation(Gs, example_ys, **kwargs)

        # reshape if necessary
        if reshaped:
            assert representation.shape[0] == 1, "Expected a single function batch dimension"
            representation = representation.squeeze(0)
        return representation, gram

    def _deterministic_inner_product(self,
                                     fs: torch.tensor,
                                     gs: torch.tensor, ) -> torch.tensor:
        """Approximates the L2 inner product between fs and gs using a Monte Carlo approximation.
        Latex: \langle f, g \rangle = \frac{1}{V}\int_X f(x)g(x) dx \approx \frac{1}{n} \sum_{i=1}^n f(x_i)g(x_i)
        Note we are scaling the L2 inner product by 1/volume, which removes volume from the monte carlo approximation.
        Since scaling an inner product is still a valid inner product, this is still an inner product.

        Args:
        fs: torch.tensor: The first set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis1)
        gs: torch.tensor: The second set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis2)

        Returns:
        torch.tensor: The inner product between fs and gs. Shape (n_functions, n_basis1, n_basis2)
        """

        # reshaping
        unsqueezed_fs, unsqueezed_gs = False, False
        if len(fs.shape) == 3:
            fs = fs.unsqueeze(-1)
            unsqueezed_fs = True
        if len(gs.shape) == 3:
            gs = gs.unsqueeze(-1)
            unsqueezed_gs = True

        # compute inner products via MC integration
        element_wise_inner_products = torch.einsum("fdmk,fdml->fdkl", fs, gs)
        inner_product = torch.mean(element_wise_inner_products, dim=1)

        # undo reshaping
        if unsqueezed_fs:
            inner_product = inner_product.squeeze(-2)
        if unsqueezed_gs:
            inner_product = inner_product.squeeze(-1)
        return inner_product

    def _inner_product(self,
                       fs: torch.tensor,
                       gs: torch.tensor) -> torch.tensor:
        """ Computes the inner product between fs and gs. This passes the data to either the deterministic or stochastic inner product methods.

        Args:
        fs: torch.tensor: The first set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis1)
        gs: torch.tensor: The second set of function outputs. Shape (n_functions, n_datapoints, input_size, n_basis2)

        Returns:
        torch.tensor: The inner product between fs and gs. Shape (n_functions, n_basis1, n_basis2)
        """

        return self._deterministic_inner_product(fs, gs)

    def _norm(self, fs: torch.tensor, squared=False) -> torch.tensor:
        """ Computes the norm of fs according to the chosen inner product.

        Args:
        fs: torch.tensor: The function outputs. Shape can vary, but typically (n_functions, n_datapoints, input_size)

        Returns:
        torch.tensor: The Hilbert norm of fs.
        """
        norm_squared = self._inner_product(fs, fs)
        if not squared:
            return norm_squared.sqrt()
        else:
            return norm_squared

    def _distance(self, fs: torch.tensor, gs: torch.tensor, squared=False) -> torch.tensor:
        """ Computes the distance between fs and gs according to the chosen inner product.

        Args:
        fs: torch.tensor: The first set of function outputs. Shape can vary, but typically (n_functions, n_datapoints, input_size)
        gs: torch.tensor: The second set of function outputs. Shape can vary, but typically (n_functions, n_datapoints, input_size)
        returns:
        torch.tensor: The distance between fs and gs.
        """
        return self._norm(fs - gs, squared=squared)

    def _compute_inner_product_representation(self,
                                              Gs: torch.tensor,
                                              example_ys: torch.tensor) -> torch.tensor:
        """ Computes the coefficients via the inner product method.

        Args:
        Gs: torch.tensor: The basis functions. Shape (n_functions, n_datapoints, output_size, n_basis)
        example_ys: torch.tensor: The output data. Shape (n_functions, n_datapoints, output_size)

        Returns:
        torch.tensor: The coefficients of the basis functions. Shape (n_functions, n_basis)
        """

        # take inner product with Gs, example_ys
        inner_products = self._inner_product(Gs, example_ys)
        return inner_products

    def _compute_least_squares_representation(self,
                                              Gs: torch.tensor,
                                              example_ys: torch.tensor,
                                              lambd: Union[float, type(None)] = None) -> Tuple[
        torch.tensor, torch.tensor]:
        """ Computes the coefficients via the least squares method.

        Args:
        Gs: torch.tensor: The basis functions. Shape (n_functions, n_datapoints, output_size, n_basis)
        example_ys: torch.tensor: The output data. Shape (n_functions, n_datapoints, output_size)
        lambd: float: The regularization parameter. None by default. If None, scales with 1/n_datapoints.

        Returns:
        torch.tensor: The coefficients of the basis functions. Shape (n_functions, n_basis)
        torch.tensor: The gram matrix. Shape (n_functions, n_basis, n_basis)
        """

        assert lambd is None or lambd >= 0, f"Expected lambda to be non-negative or None, got {lambd}"

        # set lambd to decrease with more data
        if lambd is None:
            lambd = 1e-3  # emprically this does well. We need to investigate if there is an optimal value here.

        # compute gram
        gram = self._inner_product(Gs, Gs)
        gram_reg = gram + lambd * torch.eye(gram.shape[-1], device=gram.device)

        # compute the matrix G^TF
        ip_representation = self._inner_product(Gs, example_ys)

        # Compute (G^TG)^-1 G^TF
        ls_representation = torch.einsum("fkl,fl->fk", gram_reg.inverse(),
                                         ip_representation)  # this is just batch matrix multiplication
        return ls_representation, gram

    def predict(self,
                query_xs: torch.tensor,
                representations: torch.tensor,
                precomputed_average_ys: Union[torch.tensor, None] = None) -> torch.tensor:
        """ Predicts the output of the function encoder given the input data and the coefficients of the basis functions. Uses the average function if it exists.

        Args:
        xs: torch.tensor: The input data. Shape (n_functions, n_datapoints, input_size)
        representations: torch.tensor: The coefficients of the basis functions. Shape (n_functions, n_basis)
        precomputed_average_ys: Union[torch.tensor, None]: The average function output. If None, computes it. Shape (n_functions, n_datapoints, output_size)

        Returns:
        torch.tensor: The predicted output. Shape (n_functions, n_datapoints, output_size)
        """

        # this is weighted combination of basis functions
        Gs = self.forward_basis_functions(query_xs)
        y_hats = torch.einsum("fdmk,fk->fdm", Gs, representations)

        # adds the average function prediction.
        if self.average_function is not None:
            average_ys = self.forward_average_function(query_xs)
            y_hats = y_hats + average_ys

        return y_hats

    def predict_from_examples(self,
                              example_xs: torch.tensor,
                              example_ys: torch.tensor,
                              query_xs: torch.tensor,
                              **kwargs):
        """ Predicts the output of the function encoder given the input data and the example data. Uses the average function if it exists.

        Args:
        example_xs: torch.tensor: The example input data used to compute a representation. Shape (n_example_datapoints, input_size)
        example_ys: torch.tensor: The example output data used to compute a representation. Shape (n_example_datapoints, output_size)
        xs: torch.tensor: The input data. Shape (n_functions, n_datapoints, input_size)
        method: str: "inner_product" or "least_squares". Determines how to compute the coefficients of the basis functions.
        kwargs: dict: Additional kwargs to pass to the least squares method.

        Returns:
        torch.tensor: The predicted output. Shape (n_functions, n_datapoints, output_size)
        """
        representations, gram = self.compute_representation(example_xs, example_ys, **kwargs)
        y_hats = self.predict(query_xs, representations)
        return y_hats, gram

    def forward(self, example_xs, example_ys, query_xs, **kwargs):
        return self.predict_from_examples(example_xs, example_ys, query_xs, **kwargs)

    def forward_basis_functions(self, xs: torch.tensor) -> torch.tensor:
        """
        Forward pass of the basis functions.
        NOTE: They can either be provided as a list of basis functions to be run in sequence, or a single torch module.
        This single module can run basis functions in parallel.
        """
        if type(self.basis_functions) is torch.nn.Module:
            # run basis functions in parallel
            Gs = self.basis_functions(xs)
        else:
            # run basis functions in sequence
            outs = [model(xs) for model in self.basis_functions]
            Gs = torch.stack(outs, dim=-1)
        return Gs

    def forward_average_function(self, xs: torch.tensor) -> torch.tensor:
        """ Forward pass of the average function. """
        return self.average_function(xs)