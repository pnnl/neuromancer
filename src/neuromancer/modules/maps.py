from torch import nn
from neuromancer.component import Component
import torch


class Map(Component):
    """
    Map is a component that maps aggregated inputs in input_keys into single variable and maps it via single map (nn.Module) to one or more output keys depending on the number of output keys.

    It can model following relations via single map:

    #. `one to one <https://en.wikipedia.org/wiki/One-to-one_(data_model)>`_ - if there is single input key and single output key
    #. `one to many <https://en.wikipedia.org/wiki/One-to-many_(data_model)>`_ - if there is single input key and many output keys 
    #. `many to many <https://en.wikipedia.org/wiki/Many-to-many_(data_model)>`_ - if there are many input keys and many output keys
    #. many to one - if there are many input keys and single output key
    """
    def __init__(
        self,
        func,
        input_keys,
        output_keys,
        name=None,
    ):
        """
        :param func: (nn.Module)
        :param input_keys: (list[str])
        :param output_keys: (list[str])
        :param name:
        """
        input_keys = input_keys if isinstance(input_keys, list) else [input_keys]
        output_keys = output_keys if isinstance(output_keys, list) else [output_keys]
        super().__init__(input_keys=input_keys, output_keys=output_keys, name=name)
        self.func = func

    def forward(self, data):
        x = [data[k] for k in self.input_keys]
        out = self.func(torch.cat(x, dim=-1))
        out_dict = {
            k: v for k, v in zip(
                self.output_keys,
                out if isinstance(out, tuple) else (out,)
            )
        }
        return out_dict


class ManyToMany(Map):
    """
    ManyToMany is a component that maps aggregates all inputs in input_keys into single variable
    that is mapped via single map (nn.Module) or list of maps (List[nn.Module])
    to one or more output keys depending on the number of output keys.

    It can model following relations via multiple maps:

    #. one to one - if there is single input key and single output key, maps via single map https://en.wikipedia.org/wiki/One-to-one_(data_model)
    #. one to many - if there is single input key and many output keys, maps via that many maps as output keys https://en.wikipedia.org/wiki/One-to-many_(data_model)
    #. many to many - if there are many input keys and many output keys, maps via that many maps as output keys https://en.wikipedia.org/wiki/Many-to-many_(data_model)
    #. many to one - if there are many input keys and single output key, maps via single map
    """
    def __init__(
        self,
        func,
        input_keys,
        output_keys,
        name=None,
    ):
        """
        :param func: (nn.Module) or (List[nn.Module])
        :param input_keys: (list[str])
        :param output_keys: (list[str])
        :param name:
        """
        super().__init__(func=func, input_keys=input_keys, output_keys=output_keys, name=name)
        assert isinstance(func, nn.Module) or isinstance(func, list), \
            f'map {self.func} must be either a list of nn.Modules or nn.Module'
        if isinstance(func, list):
            self.func = nn.ModuleList(func)
            assert len(self.func) == len(self.output_keys), \
                f'Number of maps {len(self.func)} must equal to number of output keys ' \
                f'{len(self.output_keys)}'
        elif isinstance(func, nn.Module):
            self.func = nn.ModuleList([func for k in range(len(self.output_keys))])

    def forward(self, data):
        out_dict = {}
        x = [data[k] for k in self.input_keys]
        x = torch.cat(x, dim=-1)
        for func, key in zip(self.func, self.output_keys):
            out_dict[key] = func(x)
        return out_dict


class OneToOne(ManyToMany):
    """
    OneToOne is a component that maps each input in input_keys exactly to one output in output_keys
    via list of maps (List[nn.Module]) with the same length as input_keys and output_keys https://en.wikipedia.org/wiki/One-to-one_(data_model)
    """
    def __init__(
        self,
        func,
        input_keys,
        output_keys,
        name=None,
    ):
        """
        :param func: (nn.Module) or (List[nn.Module])
        :param input_keys: (list[str])
        :param output_keys: (list[str])
        :param name:
        """
        super().__init__(func=func, input_keys=input_keys, output_keys=output_keys, name=name)
        assert len(self.func) == len(self.input_keys), \
            f'Number of maps {len(self.func)} must equal to number of input keys ' \
            f'{len(self.input_keys)}'

    def forward(self, data):
        out_dict = {}
        for func, in_key, out_key in zip(self.func, self.input_keys, self.output_keys):
            out_dict[out_key] = func(data[in_key])
        return out_dict


class Map2Dto3D(Map):
    """
    Map reshaping 2D tensor returned by callable func to 3D tensor based on dimensions given in
    dim1 and dim2.
    * 2D tensor: [batch, state_dim]
    * 3D tensor: [-1, dim1, dim2]
    """
    def __init__(
        self,
        func,
        input_keys,
        output_keys,
        dim1,
        dim2,
        name=None,
    ):
        """

        :param func: (nn.Module)
        :param input_keys: (list[str])
        :param output_keys: ([str])
        :param dim1: (int)
        :param dim2: (int)
        :param name:
        """
        super().__init__(func=func, input_keys=input_keys, output_keys=output_keys, name=name)
        self.dim1 = dim1
        self.dim2 = dim2

        assert self.func.out_features == dim1*dim2, \
            f'product of dim1 {dim1} and dim2 {dim2} must equal to func.out_features {self.func.out_features}'

    def forward(self, data):
        """

        :param data: (dict {str: torch.tensor)} 2D tensor
        :return: (dict {str: torch.tensor)} 3D tensor
        """
        out_dict = {}
        x = [data[k] for k in self.input_keys]
        out_2D = self.func(torch.cat(x, dim=-1))
        out_3D = out_2D.reshape(-1, self.dim1, self.dim2)
        out_dict[self.output_keys[0]] = out_3D
        return out_dict


class Transform(Component):
    def __init__(
        self,
        callable,
        input_keys,
        output_keys=[],
        name=None,
    ):
        """
        perform any pytorch transformation via callable

        :param callable: (callable)
        :param input_keys: (list[str])
        :param output_keys: (list[str])
        :param name:
        """
        input_keys = input_keys if isinstance(input_keys, list) else [input_keys]
        output_keys = output_keys if isinstance(output_keys, list) else [output_keys]
        if bool(output_keys):
            assert len(input_keys) == len(output_keys), \
                f'Number of input keys {len(input_keys)} ' \
                f'must equal to number of output keys {len(output_keys)} '
        else:
            output_keys = input_keys
        super().__init__(input_keys=input_keys, output_keys=output_keys, name=name)
        self.callable = callable

    def forward(self, data):
        out_dict = {}
        for in_key, out_key in zip(self.input_keys, self.output_keys):
            out_dict[out_key] = self.callable(data[in_key])
        return out_dict