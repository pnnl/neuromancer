
from typing import List
from abc import ABC, abstractmethod
from torch import nn
from neuromancer.component import Component
import torch


class Map(Component):
    def __init__(
        self,
        func,
        input_keys,
        output_keys,
        name=None,
    ):
        """
        Map is a component that maps aggregated inputs in input_keys into single variable
        and maps it via single map (nn.Module) to one or more output keys depending on the number of output keys.

        It can model following relations via single map:
            one to one - if there is single input key and single output key
                https://en.wikipedia.org/wiki/One-to-one_(data_model)
            one to many - if there is single input key and many output keys
                https://en.wikipedia.org/wiki/One-to-many_(data_model)
            many to one - if there are many input keys and single output key
            many to many - if there are many input keys and many output keys
                https://en.wikipedia.org/wiki/Many-to-many_(data_model)

        :param map: (nn.Module)
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
    def __init__(
        self,
        func,
        input_keys,
        output_keys,
        name=None,
    ):
        """
        ManyToMany is a component that maps aggregates all inputs in input_keys into single variable
        that is mapped via single map (nn.Module) or list of maps (List[nn.Module])
        to one or more output keys depending on the number of output keys.

        It can model following relations via multiple maps:
            one to one - if there is single input key and single output key, maps via single map
                https://en.wikipedia.org/wiki/One-to-one_(data_model)
            one to many - if there is single input key and many output keys, maps via that many maps as output keys
                https://en.wikipedia.org/wiki/One-to-many_(data_model)
            many to many - if there are many input keys and many output keys, maps via that many maps as output keys
                https://en.wikipedia.org/wiki/Many-to-many_(data_model)
            many to one - if there are many input keys and single output key, maps via single map

        :param map: (nn.Module) or (List[nn.Module])
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
    def __init__(
        self,
        func,
        input_keys,
        output_keys,
        name=None,
    ):
        """
        OneToOne is a component that maps each input in input_keys exactly to one output in output_keys
        via list of maps (List[nn.Module]) with the same length as input_keys and output_keys
            https://en.wikipedia.org/wiki/One-to-one_(data_model)

        :param map: (nn.Module) or (List[nn.Module])
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


