


#
#
# # TODO: differentiate autonomous and non-autonomous odes
# # and odes with and without disturbances
# """
# gray-box ODE for training with customly defined nonlinmap
# then use them as specifi nonlinear and linear maps
# e.g., nonlinmap = [CSTR, LorentzSystem]
# linmap = identity not being learned
# """
class GrayODE(nn.Module):
    def __init__(self, nx, nu, nd, ny, fxud, fy):
        """
        atomic gray-box ODE model
        :param nx: (int) dimension of state
        :param nu: (int) dimension of inputs
        :param nd: (int) dimension of disturbances
        :param ny: (int) dimension of observation
        :param fxud: function R^{nx+nu+nd} -> R^(nx}
        :param fy: function R^{nx} -> R^(ny}

        generic discrete-time ODE system dynamics:
        # x+ = fxud(x,u,d)
        # y =  fy(x)
        """
        super().__init__()
        assert fxud.in_features == nx+nu+nd, "Mismatch in input function size"
        assert fxud.nx == nx, "Mismatch in input function size"
        assert fxud.nu == nu, "Mismatch in input function size"
        assert fxud.nd == nd, "Mismatch in input function size"
        assert fy.in_features == nx, "Mismatch in observable output function size"
        assert fy.out_features == ny, "Mismatch in observable output function size"

        self.nx, self.nu, self.nd, self.ny = nx, nu, nd, ny
        self.fxud, self.fy = fxud, fy
        # TODO: if fy is None, then use identity without gradient

        # Regularization Initialization
        self.xmin, self.xmax, self.umin, self.umax = self.con_init()
        self.Q_dx, self.Q_con_x, self.Q_con_u, self.Q_sub = self.reg_weight_init()

        # slack variables for calculating constraints violations/ regularization error
        self.sxmin, self.sxmax, self.sumin, self.sumax, self.sdx_x, self.dx_u, self.dx_d, self.s_sub = [0.0] * 8

    def running_mean(self, mu, x, n):
        return mu + (1/n)*(x - mu)

    def reset(self):
        for mod in self.modules():
            if hasattr(mod, 'reset') and mod is not self:
                mod.reset()

    def forward(self, x, U=None, D=None, nsamples=1):
        """
        """
        if U is not None:
            nsamples = U.shape[0]
        elif D is not None:
            nsamples = D.shape[0]

        X, Y = [], []
        for i in range(nsamples):
            u = U[i] if U is not None else None
            d = D[i] if D is not None else None
            x_prev = x
            x = self.fxud(x, u, d)
            y = self.fy(x)
            X.append(x)
            Y.append(y)
            self.regularize(x_prev, x, u, i+1)
        self.reset()
        return torch.stack(X), torch.stack(Y), self.reg_error()
