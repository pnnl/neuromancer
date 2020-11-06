import torch
import dill
from neuromancer.datasets import EmulatorDataset

from lpv import lpv
from phase_plots import plot_model_phase_portrait
from eigen_plots import compute_eigenvalues, plot_Astar_anim

# TODO(lltt): add support for gershgorin tutorial system (no estimator, f_u)


def sample_data_trajectories(model, data, fname=None):
    fx = model.components[1].fx
    fu = model.components[1].fu
    estim = model.components[0]

    x = estim(data)["x0_estim"]
    A_stars = []
    for u in data["Up"]:
        A_star, _, _ = lpv(fx, x)
        x = fx(x) + fu(u)
        A_stars += [A_star.detach().cpu().numpy()]
    eigvals = compute_eigenvalues(A_stars)

    return A_stars, eigvals


def sample_random_trajectories(model, nsamples=5, nsteps=50):
    fx = model.components[1].fx
    fu = model.components[1].fu
    nx = fx.in_features
    nu = fu.in_features

    # TODO(lltt): constant inputs? step inputs (from dataset)?
    random_inputs = torch.rand(nsteps, nu)

    samples = []
    for _ in range(nsamples):
        A_stars = []
        x = torch.rand(1, nx)
        for t in range(nsteps):
            Astar, _, _ = lpv(fx, x)
            x = fx(x) + fu(random_inputs[t : t + 1, ...])
            A_stars += [Astar.detach().cpu().numpy()]

        eigvals = compute_eigenvalues(A_stars)
        samples += [(A_stars, eigvals)]

    return samples


if __name__ == "__main__":
    CSTR_MODEL_PATH = "neuromancer/train_scripts/L4DC_paper/models/cstr_model.pth"
    TANK_MODEL_PATH = "neuromancer/train_scripts/L4DC_paper/models/tank_model.pth"

    # CSTR A* visualizations
    cstr_model = torch.load(CSTR_MODEL_PATH, pickle_module=dill, map_location="cpu")
    cstr_data = EmulatorDataset("CSTR", nsim=10000, seed=50, device="cpu")

    print("CSTR: Sampling trajectories from training data...")
    A_stars, eigvals = sample_data_trajectories(cstr_model, cstr_data.train_loop)
    plot_Astar_anim(A_stars, eigvals, fname=f"cstr_sample.mp4")

    print("CSTR: Sampling trajectories from random data...")
    samples = sample_random_trajectories(cstr_model)
    for i, (A_stars, eigvals) in enumerate(samples):
        plot_Astar_anim(A_stars, eigvals, fname=f"cstr_random_sample_{i}.mp4")

    # two tank A* visualizations
    tank_data = EmulatorDataset("TwoTank", nsim=10000, seed=81, device="cpu")
    tank_model = torch.load(TANK_MODEL_PATH, pickle_module=dill, map_location="cpu")

    print("Two Tank: Sampling A* trajectories from training data...")
    A_stars, eigvals = sample_data_trajectories(tank_model, tank_data.train_loop)
    plot_Astar_anim(A_stars, eigvals, fname=f"tank_sample.mp4")

    print("Two Tank: Sampling A* trajectories from random data...")
    samples = sample_random_trajectories(tank_model)
    for i, (A_stars, eigvals) in enumerate(samples):
        plot_Astar_anim(A_stars, eigvals, fname=f"tank_random_sample_{i}.mp4")
