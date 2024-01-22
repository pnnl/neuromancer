from abc import ABC, abstractmethod
import torch


class Interpolation(ABC):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def interpolation(self, tq, t, u):
        pass

    def __call__(self, tq, t=None, u=None):
        return self.interpolation(tq, t, u)


class LinInterp_Offline(Interpolation):

    def __init__(self, t, u):
        """
        Offline interpolation for time series with fixed sampling rate

        :param t: torch.Tensor (# of timesteps, 1) time vector with fixed sampling rate and ascending order
        :param u: torch.Tensor (# of timesteps, state dim)
        """
        super().__init__()
        assert len(list(t.size())) == 2, 't should be a 2D torch tensor' 
        assert len(list(u.size())) == 2, 'u should be a 2D torch tensor'
        assert torch.diff(t).any() >= 0, 't should be ascending order'
        self.t = t
        self.u = u
        # check timestep size in self.t is identical or not
        if self.t.shape[0] == 1:
            dt_all = torch.zeros(1, 1)
        else:
            dt_all = torch.diff(self.t, dim=0).to(torch.float64)
        self.dt_mean = torch.mean(dt_all)*torch.ones_like(dt_all)

    def interpolation(self, tq, t=None, u=None):
        """
        :param tq: torch.Tensor (# of timesteps, 1),
                    The unit of tq is actual temporal unit, e.g. second, not index.
        :return: torch.Tensor (# of timesteps, state dim)
        """
        # if self.t is uniformly sampled.
        tq_ind = ((tq - self.t[0, 0])/(self.dt_mean[0, 0])).flatten()
        lower_bound = torch.Tensor.int(torch.floor(tq_ind)).to(torch.int64)
        upper_bound = torch.Tensor.int(torch.ceil(tq_ind)).to(torch.int64)
        # check if the max of tq is greater than that of self.t
        ind_max = self.u.shape[0] - 1
        if torch.amax(upper_bound) > ind_max:
            ind_extrap = torch.nonzero(upper_bound > ind_max)
            lower_bound[ind_extrap] = ind_max - 1
            upper_bound[ind_extrap] = ind_max
        # check if the min of tq is smaller than that of self.t
        ind_min = 0
        if torch.amin(lower_bound) < ind_min:
            ind_extrap = torch.nonzero(lower_bound < ind_min)
            lower_bound[ind_extrap] = ind_min
            upper_bound[ind_extrap] = ind_min + 1
        distance = (tq_ind - lower_bound).unsqueeze(-1)
        uq = distance*(self.u[upper_bound, :] - self.u[lower_bound, :]) + self.u[lower_bound, :]
        return uq.float()


class LinInterp_Online(Interpolation):

    def __init__(self):
        """
        Online linear interpolation class
        """
        super().__init__()

    def interpolation(self, tq, t, u):
        # u (2, batch #, state dim)
        # t (2, batch #, 1)
        # tq (2, batch #, 1)
        tq = tq[:, 0, :]  # though imported tq is 3D, we only use its first slice
        uq = (tq - t[:, 0, :])/(t[:, 1, :] - t[:, 0, :])*(u[:, 1, :] - u[:, 0, :]) + u[:, 1, :]
        return uq  # (batch #, state dim)

