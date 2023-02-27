from torch import nn


def _validate_key_params(keys):
    return keys is None or isinstance(keys, dict) or isinstance(keys, list)


def check_key_subset(k1, k2):
    """
    Check that keys in k1 are a subset of keys in k2

    :param k1: iterable of str
    :param k2: iterable of str
    """
    common_keys = set(k1) - set(k2)

    assert len(common_keys) == 0, \
            f'Missing values in dataset: {common_keys}\n' \
            f'  input_keys: {set(k1)}\n  data_keys: {set(k2)}'


class Component(nn.Module):

    def __init__(self, input_keys, output_keys, name=None):
        """
        The NeuroMANCER component base class.

        :param input_keys: (List of str) list of strings representing keys (names) of input variables.
        :param output_keys: (List of str) list of strings representing keys (names) of output variables.
        :param name: (str) The name of the component, used to tag output variable keys with the name
            of the component that produced them.
        """
        super().__init__()
        self.input_keys = input_keys
        self.output_keys = output_keys
        self.name = name
        self.register_forward_pre_hook(self._check_inputs)
        self.register_forward_hook(self._check_output)

    def _check_inputs(self, module, input_data):

        input_data = input_data[0]
        set_diff = set(self.input_keys) - set(input_data)
        keys = [k for k in input_data.keys()]
        assert len(set_diff) == 0, \
            f" Missing inputs {set_diff} for component {module.name}, only got {keys}"

    def _check_output(self, module, input_data, output_data):
        assert set(output_data.keys()) == set(self.output_keys), \
            f' Key mismatch \n' \
            f' Forward pass keys: {set(output_data.keys())} \n' \
            f' Default output keys: {set(self.output_keys)}'

    def __repr__(self):
        return f"{self.name}({', '.join(self.input_keys)}) -> {', '.join(self.output_keys)}"

