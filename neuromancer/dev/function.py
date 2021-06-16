from neuromancer.component import Component


class Function(Component):
    def __init__(
        self,
        func,
        input_keys,
        output_key,
        name,
    ):
        self.DEFAULT_OUTPUT_KEYS = [output_key]
        super().__init__(
            input_keys,
            [output_key],
            name,
        )
        self.func = func
        self.output_key = output_key
        
    def forward(self, data):
        x = [data[k] for k in self.input_keys]
        return {self.output_key: self.func(*x)}


if __name__ == "__main__":
    import torch
    from torch import nn
    from neuromancer.datasets import DataDict
    from neuromancer.dynamics import _bssm_kinds, _extract_dims
    from neuromancer.problem import Problem
    from neuromancer import blocks
    import slim

    test_data = DataDict({
        "x0": torch.rand(20, 5),
        "Yp": torch.rand(10, 20, 5),
        "Yf": torch.rand(10, 20, 5),
        "Uf": torch.rand(10, 20, 3),
        "Df": torch.rand(10, 20, 2),
    })
    test_data.name = "test"
    data_dims = {k: v.shape[-1:] for k, v in test_data.items()}

    def block_model(
        kind,
        datadims,
        linmap,
        nonlinmap,
        bias,
        n_layers=2,
        fe=None,
        fyu=None,
        activation=nn.GELU,
        residual=False,
        linargs=dict(),
        timedelay=0,
        xou=torch.add,
        xod=torch.add,
        xoe=torch.add,
        xoyu=torch.add,
        name='blockmodel',
        input_keys=dict()
    ):
        """
        Generate a block-structured SSM with the same structure used across fx, fy, fu, and fd.
        """
        assert kind in _bssm_kinds, \
            f"Unrecognized model kind {kind}; supported models are {_bssm_kinds}"

        nx, ny, nu, nd, nx_td, nu_td, nd_td = _extract_dims(datadims, input_keys, timedelay)
        hsizes = [nx] * n_layers

        lin = lambda ni, no: (
            linmap(ni, no, bias=bias, linargs=linargs)
        )
        nlin = lambda ni, no: (
            nonlinmap(ni, no, bias=bias, hsizes=hsizes, linear_map=linmap, nonlin=activation, linargs=linargs)
        )

        # define (non)linearity of each component according to given model type
        if kind == "blocknlin":
            fx = nlin(nx_td, nx)
            fy = lin(nx_td, ny)
            fu = nlin(nu_td, nx) if nu != 0 else None
            fd = nlin(nd_td, nx) if nd != 0 else None
        elif kind == "linear":
            fx = lin(nx_td, nx)
            fy = lin(nx_td, ny)
            fu = lin(nu_td, nx) if nu != 0 else None
            fd = lin(nd_td, nx) if nd != 0 else None
        elif kind == "hammerstein":
            fx = lin(nx_td, nx)
            fy = lin(nx_td, ny)
            fu = nlin(nu_td, nx) if nu != 0 else None
            fd = nlin(nd_td, nx) if nd != 0 else None
        elif kind == "weiner":
            fx = lin(nx_td, nx)
            fy = nlin(nx_td, ny)
            fu = lin(nu_td, nx) if nu != 0 else None
            fd = lin(nd_td, nx) if nd != 0 else None
        else:   # hw
            fx = lin(nx_td, nx)
            fy = nlin(nx_td, ny)
            fu = nlin(nu_td, nx) if nu != 0 else None
            fd = nlin(nd_td, nx) if nd != 0 else None

        subgraph = []
        fx_inputs = [input_keys["x0"], input_keys["Yf"]]
        if fu is not None:
            fu = Function(fu, [input_keys["Uf"]], "out", "fu")
            subgraph.append(fu)
            fx_inputs.append("out_fu")

        if fd is not None:
            fd = Function(fd, [input_keys["Df"]], "out", "fd")
            subgraph.append(fd)
            fx_inputs.append("out_fd")

        """
        # TODO: currently not supported
        fe = (
            Function(
                fe(nx_td, nx, hsizes=hsizes, bias=bias, linear_map=linmap, nonlin=activation, linargs=dict()),
                ["X_pred_fx"],
                "out",
                "fe"
            )
            if kind in {"blocknlin", "hammerstein", "hw"}
            else Function(fe(nx_td, nx, bias=bias, linargs=linargs), ["X_pred_fx"], "out", "fe")
        ) if fe is not None else None

        fyu = (
            fyu(nu_td, ny, hsizes=hsizes, bias=bias, linear_map=linmap, nonlin=activation, linargs=dict())
            if kind in {"blocknlin", "hw"}
            else fyu(nu_td, ny, bias=bias, linargs=linargs)
        ) if fyu is not None else None
        """

        def fx_iterator(fx, residual=False):
            def closure(x, yf, *args):
                steps = yf.shape[0]
                states = x.new_zeros(steps, *x.shape)
                for i in range(steps):
                    x_prev = x
                    x = fx(x)
                    for a in args:
                        x = x + a[i]
                    x = x + x_prev * int(residual)
                    states[i] = x
                return states
            return closure

        fx = Function(
            fx_iterator(fx, residual),
            fx_inputs,
            "X_pred",
            "fx"
        )
        fy = Function(fy, ["X_pred_fx"], "Y_pred", "fy")

        return [*subgraph, fx, fy]

    components = block_model(
        "hw",
        data_dims,
        slim.Linear,
        blocks.MLP,
        False,
        n_layers=2,
        fe=None,
        fyu=None,
        activation=nn.GELU,
        residual=True,
        linargs=dict(),
        timedelay=0,
        xou=torch.add,
        xod=torch.add,
        xoe=torch.add,
        xoyu=torch.add,
        name='blockmodel',
        input_keys={
            "x0": "x0",
            "Uf": "Uf",
            "Yf": "Yf",
            "Df": "Df",
        },
    )
    model = Problem([], [], components)
    print(model)

    out = model(test_data)
    print(out.keys())