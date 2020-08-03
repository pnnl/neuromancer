
# TODO: implement multistep neural network - exactly copy their implementation for comparison
# https://arxiv.org/abs/1801.01236
# https://github.com/maziarraissi/MultistepNNs/blob/master/Multistep_NN.py


# # TODO: finish and test our version of multistep method
# # instead of fixed values of parameters we will learn them
class MultistepBlack(BlackSSM):
    def __init__(self, nx, nu, nd, ny, fxud, fy, M=1):
        """
        Implements generic explicit multistep formula of Adams–Bashforth methods
        x_k = x_k-1 + h*(b_1*f(x_k-1) +  b_2*f(x_k-2) + ... b_M*f(x_k-M))
        sum(b_i) = 1, for all i = 1,...,M
        h = step size

        # TODO: implement stability condition on coefficients
        A linear multistep method is zero-stable if and only if the root condition is satisfied
        # https://en.wikipedia.org/wiki/Linear_multistep_method
        """
        super().__init__(nx, nu, nd, ny, fxud, fy)
        self.M = M  # number of steps of the multistep method
        self.h = 1  # step size, default = 1
        self.Bsum_error = 0.0 # regularization error of the multistep coefficient constraints

    # TODO: move this formulat to blocks?
    def AdamsBashforth(self, X, U, D, M):
        # computing Adams Bashforth residual
        self.Beta = nn.Parameter(torch.randn(1, M))
        for j in range(M):
            u = None
            x = X[j]
            xi = x
            if U is not None:
                u = U[j]
                xi = torch.cat([xi, u], dim=1)
            if D is not None:
                d = D[j]
                xi = torch.cat([xi, d], dim=1)
            dx = self.h*(self.Beta[j]*self.fxud(xi)) if j == 0 \
                else dx + self.h*(self.Beta[j]*self.fxud(xi))
        x = x + dx
        self.Bsum_error = (1-sum(self.Beta))*(1-sum(self.Beta))
        return x, u

    def forward(self, x, U=None, D=None, nsamples=1):
        """
        """
        if U is not None:
            nsamples = U.shape[0]
        elif D is not None:
            nsamples = D.shape[0]

        X, Y = [], []

        for i in range(nsamples):
            x_prev = x
            if i == 0:
                Mi = 0
                Xi = x
                Ui = U[i] if U is not None else None
                Di = D[i] if D is not None else None
            elif i<self.M:
                Mi = i
                Xi = Xi.append(x)
                Ui = U[0:i] if U is not None else None
                Di = D[0:i] if D is not None else None
            else:
                Mi = self.M
                Xi = Xi.append(x)
                Xi = Xi[i-self.M:i]
                Ui = U[i-self.M:i] if U is not None else None
                Di = D[i-self.M:i] if D is not None else None
            x, u = self.AdamsBashforth(Xi, Ui, Di, Mi)
            y = self.fy(x)
            X.append(x)
            Y.append(y)
            self.regularize(x_prev, x, u, i + 1)

        self.reset()
        return torch.stack(X), torch.stack(Y), self.reg_error()+self.Bsum_error



