"""
Standalone Python script equivalent to Part_7_SFNO_ShallowWater.ipynb.

Mirrors the notebook workflow end-to-end, including plotting and training.
"""

import warnings
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from neuralop.data.datasets import load_spherical_swe
from neuralop.utils import count_model_params
from neuralop import LpLoss

# Enable local neuromancer source when running from the repository root
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from neuromancer.dataset import DictDataset
from neuromancer.modules.operators import SFNO
from neuromancer.system import Node
from neuromancer.constraint import Loss
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer


PLOTS_DIR = Path(__file__).resolve().parent / "plots"


def save_fig(name: str):
    """Save the current matplotlib figure to the plots directory and close it."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / name)
    plt.close()


def select_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def unpack_swe_sample(sample):
    """Return (x, y) as [C, H, W] tensors."""
    x = sample["x"]
    y = sample["y"]
    if y.ndim == 4:
        y = y[0]
    if x.ndim == 4:
        x = x[0]
    return x.float(), y.float()


def stack_swe_dataset(dataset):
    """Stack the full dataset into [N, C, H, W] tensors."""
    xs, ys = [], []
    for i in range(len(dataset)):
        x, y = unpack_swe_sample(dataset[i])
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)


def plot_swe_fields(x, y, title, filename):
    """Visualize input and target channels side by side."""
    x = x.detach().cpu()
    y = y.detach().cpu()
    if x.ndim == 2:
        x = x.unsqueeze(0)
    if y.ndim == 2:
        y = y.unsqueeze(0)

    n_channels = min(x.shape[0], y.shape[0], 3)
    field_names = ["u", "v", "h"]

    fig = plt.figure(figsize=(4 * n_channels, 6))
    for i in range(n_channels):
        ax = fig.add_subplot(2, n_channels, i + 1)
        ax.imshow(x[i])
        ax.set_title(f"Input {field_names[i] if i < len(field_names) else f'ch{i}'}")
        ax.set_xticks([])
        ax.set_yticks([])

        ax = fig.add_subplot(2, n_channels, n_channels + i + 1)
        ax.imshow(y[i])
        ax.set_title(f"Target {field_names[i] if i < len(field_names) else f'ch{i}'}")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, y=0.98)
    save_fig(filename)


def plot_prediction(x, y, y_pred, title, filename, channel=0):
    """Plot a single channel prediction against ground truth."""
    x = x.detach().cpu()
    y = y.detach().cpu()
    y_pred = y_pred.detach().cpu()

    if x.ndim == 2:
        x = x.unsqueeze(0)
    if y.ndim == 2:
        y = y.unsqueeze(0)
    if y_pred.ndim == 2:
        y_pred = y_pred.unsqueeze(0)

    u_true = y[channel].numpy()
    u_pred = y_pred[channel].numpy()
    denom = np.linalg.norm(u_true)
    rel_l2 = np.linalg.norm(u_true - u_pred) / denom if denom != 0 else float("nan")
    print(f"Relative L2 error (channel {channel}): {rel_l2:.4e}")

    vmin = u_true.min()
    vmax = u_true.max()

    fig = plt.figure(figsize=(12, 4))
    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(x[channel])
    ax.set_title("Input")
    ax = fig.add_subplot(1, 3, 2)
    ax.imshow(y[channel], vmin=vmin, vmax=vmax)
    ax.set_title("Ground truth")
    ax = fig.add_subplot(1, 3, 3)
    ax.imshow(y_pred[channel], vmin=vmin, vmax=vmax)
    ax.set_title(f"SFNO prediction (rel L2: {rel_l2:.3e})")

    for ax in fig.axes:
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(title, y=0.98)
    save_fig(filename)


def main():
    warnings.filterwarnings("ignore")
    torch.set_default_dtype(torch.float)
    torch.manual_seed(1234)
    np.random.seed(1234)

    device = select_device()
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------
    n_train = 200
    batch_size_train = 32
    train_resolution = (32, 64)

    n_tests = [40, 40]
    test_resolutions = [(32, 64), (64, 128)]
    test_batch_sizes = [40, 40]

    train_loader, test_loaders = load_spherical_swe(
        n_train=n_train,
        batch_size=batch_size_train,
        train_resolution=train_resolution,
        test_resolutions=test_resolutions,
        n_tests=n_tests,
        test_batch_sizes=test_batch_sizes,
    )

    train_dataset = train_loader.dataset
    print(f"Train dataset size: {len(train_dataset)}")

    sample = train_dataset[0]
    print(sample.keys())
    print("x shape:", sample["x"].shape)
    print("y shape:", sample["y"].shape)

    vis_x, vis_y = unpack_swe_sample(train_dataset[0])
    plot_swe_fields(
        vis_x, vis_y, "Spherical SWE training sample", "sfno_swe_sample.png"
    )

    # ------------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------------
    x_train, y_train = stack_swe_dataset(train_dataset)
    x_train = x_train.to(device)
    y_train = y_train.to(device)

    train_ds = DictDataset({"x_grid": x_train, "y_grid": y_train}, name="train")
    train_loader_nm = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size_train,
        shuffle=True,
        collate_fn=train_ds.collate_fn,
    )

    test_ds_by_res = {}
    for res in test_resolutions:
        test_dataset = test_loaders[res].dataset
        x_test, y_test = stack_swe_dataset(test_dataset)
        x_test = x_test.to(device)
        y_test = y_test.to(device)
        test_ds_by_res[res] = DictDataset(
            {"x_grid": x_test, "y_grid": y_test},
            name=f"test_{res[0]}x{res[1]}",
        )

    print("Train x:", train_ds.datadict["x_grid"].shape)
    print("Train y:", train_ds.datadict["y_grid"].shape)
    for res, ds in test_ds_by_res.items():
        print(f"Test {res} x:", ds.datadict["x_grid"].shape)
        print(f"Test {res} y:", ds.datadict["y_grid"].shape)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    in_channels = x_train.shape[1]
    out_channels = y_train.shape[1]
    n_modes = (train_resolution[0] // 2, train_resolution[1] // 2)

    sfno_model = SFNO(
        n_modes=n_modes,
        in_channels=in_channels,
        out_channels=out_channels,
        hidden_channels=64,
        domain_padding=[0.05, 0.05],
        n_layers=2,
    ).to(device)

    n_params = count_model_params(sfno_model)
    print(f"Our model has {n_params} parameters.")

    sfno_node = Node(sfno_model, ["x_grid"], ["y_hat"], name="sfno_model")
    print("symbolic inputs:", sfno_node.input_keys)
    print("symbolic outputs:", sfno_node.output_keys)

    with torch.no_grad():
        sample_out = sfno_node({"x_grid": x_train[:1]})
    print("SFNO output shape:", sample_out["y_hat"].shape)

    # ------------------------------------------------------------------
    # Losses and problem
    # ------------------------------------------------------------------
    l2loss = LpLoss(d=2, p=2, reduction="sum")

    l2_obj = Loss(
        ["y_hat", "y_grid"],
        lambda yhat, y: l2loss(yhat, y),
        name="l2_loss",
    )

    loss = PenaltyLoss([l2_obj], [])
    problem = Problem(nodes=[sfno_node], loss=loss)
    problem.show()

    # ------------------------------------------------------------------
    # Trainer
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(problem.parameters(), lr=5e-3, weight_decay=1e-4)
    epochs = 30

    trainer = Trainer(
        problem.to(device),
        train_loader_nm,
        optimizer=optimizer,
        epochs=epochs,
        epoch_verbose=5,
        train_metric="train_loss",
        dev_metric="train_loss",
        eval_metric="train_loss",
        warmup=epochs,
        device=device,
    )

    best_model = trainer.train()
    problem.load_state_dict(best_model)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    sfno_eval = problem.nodes[0].to(device)

    for res, test_ds in test_ds_by_res.items():
        x_test = test_ds.datadict["x_grid"]
        y_test = test_ds.datadict["y_grid"]
        with torch.no_grad():
            preds = sfno_eval({"x_grid": x_test})["y_hat"]
        plot_prediction(
            x_test[0],
            y_test[0],
            preds[0],
            title=f"SFNO prediction at resolution {res}",
            filename=f"sfno_swe_pred_{res[0]}x{res[1]}.png",
            channel=0,
        )


if __name__ == "__main__":
    main()
