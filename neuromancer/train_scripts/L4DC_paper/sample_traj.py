import torch
import dill
from neuromancer.datasets import EmulatorDataset
from tqdm import tqdm

from lpv import lpv
from phase_plots import plot_random_phase_portrait, plot_phase_portrait_2d
from eigen_plots import compute_eigenvalues, plot_Astar_anim

# TODO(lltt):
# - focus on no input case first, then extrapolate from there
# - projection from high-dimensional state space:
#   - random projection?


def sample_data_trajectories(model, data):
    fx = model.components[1].fx
    fu = model.components[1].fu
    estim = model.components[0]

    loop_data = data.train_loop

    x = estim(loop_data)["x0_estim"]
    A_stars = []
    for u in loop_data["Up"]:
        A_star, _, _ = lpv(fx, x)
        x = torch.matmul(x, A_star) + fu(u)
        A_stars += [A_star.detach().cpu().numpy()]
    eigvals = compute_eigenvalues(A_stars)

    plot_Astar_anim(A_stars, eigvals, fname=f"{system}_sample.mp4")


def sample_random_trajectories(model, nsamples=1000, nsteps=100):
    fx = model.components[1].fx
    fu = model.components[1].fu
    nx = fx.in_features
    nu = fu.in_features

    random_inputs = torch.rand(nsteps, nu, device="cuda:0")

    samples = []
    for i in tqdm(range(nsamples)):
        A_stars = []
        x = torch.rand(1, nx, dtype=torch.float, device="cuda:0")
        for t in tqdm(range(nsteps)):
            Astar, _, _ = lpv(fx, x)
            x = torch.matmul(x, Astar) + fu(random_inputs[t : t + 1, ...])
            A_stars += [Astar.detach().cpu().numpy()]

        eigvals = compute_eigenvalues(A_stars)

        samples += [(x, A_stars, eigvals)]

    plot_Astar_anim(A_stars, eigvals, fname=f"sample_{i}.mp4")

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # TODO(lltt): TwoTank model is NOT a block nonlinear model, needs nonlinear state transition
    CSTR_MODEL_PATH = "neuromancer/train_scripts/L4DC_paper/models/best_model_cstr.pth"
    TANK_MODEL_PATH = "neuromancer/train_scripts/ACC_models/best_models_bnl_paper/blocknlin/twotank/best_model.pth"

    cstr_model = torch.load(CSTR_MODEL_PATH, pickle_module=dill, map_location="cpu")
    cstr_data = EmulatorDataset("CSTR", nsim=10000, seed=50, device="cuda:0")

    tank_model = torch.load(TANK_MODEL_PATH, pickle_module=dill, map_location="cpu")

    plot_phase_portrait_2d(cstr_model, data=cstr_data.train_loop)
    plt.show()

    plot_phase_portrait_2d(tank_model)
    plt.show()

    sample_data_trajectories(cstr_model, "CSTR")
    sample_random_trajectories(cstr_model)

    sample_random_trajectories(tank_model)
    # sample_data_trajectories(tank_model, "TwoTank")
