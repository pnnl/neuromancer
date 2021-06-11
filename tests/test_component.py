from neuromancer.component import Component


class DummyComponent(Component):
    DEFAULT_INPUT_KEYS = ["X", "Y"]
    DEFAULT_OUTPUT_KEYS = ["X_pred", "Y_pred", "reg_error"]

    def __init__(
        self,
        input_keys=None,
        output_keys=None,
        name="dummy",
    ):
        super().__init__(
            input_keys=input_keys or DummyComponent.DEFAULT_INPUT_KEYS,
            output_keys=output_keys or DummyComponent.DEFAULT_OUTPUT_KEYS,
            name=name,
        )

    def forward(self, data):
        return {
            "X_pred": data["X"],
            "Y_pred": data["Y"],
            "reg_error": 13.0,
        }


def test_input_key_list():
    input_keys = ["X", "Y", "Z"]
    component = DummyComponent(
        input_keys=input_keys,
        name="input_list_test",
    )

    data = {"X": 123, "Y": 321}

    out = component(data)


def test_input_remap():
    remapped_keys = {"X_renamed": "X", "Y_renamed": "Y"}
    component = DummyComponent(
        input_keys=remapped_keys,
        name="input_remap_test",
    )

    data = {"X_renamed": 123, "Y_renamed": 321}
    out = component._remap_input(component, (data,))

    assert all([k in remapped_keys.values() for k in out.keys()])


def test_output_remap():
    remapped_keys = {"X_pred": "X_renamed", "Y_pred": "Y_renamed"}
    component = DummyComponent(
        output_keys=remapped_keys,
        name="output_remap_test",
    )

    data = {"X_pred": 123, "Y_pred": 321}
    out = component._remap_output(component, {}, data)

    print(out)

    assert all([k in component.output_keys for k in out.keys()])


def test_forward_remap():
    remapped_input = {"X_renamed": "X", "Y_renamed": "Y"}
    remapped_output = {"X_pred": "X_renamed", "Y_pred": "Y_renamed"}
    component = DummyComponent(
        input_keys=remapped_input,
        output_keys=remapped_output,
        name="full_remap_test",
    )

    data = {"X_renamed": 123, "Y_renamed": 321}
    out = component(data)

    assert all([k in component.output_keys for k in out.keys()])
