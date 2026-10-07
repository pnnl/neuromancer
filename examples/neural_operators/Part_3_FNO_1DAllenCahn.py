"""
Standalone Python script equivalent to Part_3_FNO_1DAllenCahn.ipynb.

Trains a 1D Fourier Neural Operator on Allen-Cahn data with Neuromancer and saves
plots and downloaded data locally.
"""

import sys
import time
import urllib.request
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

# Keep torch warnings quieter
warnings.filterwarnings("ignore")

# Enable local neuromancer source when running from the repository root
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root / "src"))

from neuromancer.callbacks import LossHistoryCallback
from neuromancer.constraint import variable
from neuromancer.dataset import DictDataset
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.system import Node
from neuromancer.trainer import Trainer


PLOTS_DIR = Path(__file__).resolve().parent / "plots"
DATA_DIR = Path(__file__).resolve().parent / "data"


def save_fig(name: str):
    """Save the current matplotlib figure to the plots directory and close it."""
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / name)
    plt.close()


def choose_device() -> torch.device:
    """Select the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def download_data():
    """Download Allen-Cahn dataset if missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    base = (
        "https://raw.githubusercontent.com/camlab-ethz/AI_Science_Engineering/main/datasets"
    )
    for fname in ("AC_data_input.npy", "AC_data_output.npy"):
        url = f"{base}/{fname}"
        dst = DATA_DIR / fname
        if not dst.exists():
            print(f"Downloading {fname} to {dst}...")
            urllib.request.urlretrieve(url, dst)


def load_data():
    """Load Allen-Cahn tensors, downloading first if needed."""
    download_data()
    x_data = torch.from_numpy(np.load(DATA_DIR / "AC_data_input.npy")).float()
    y_data = (
        torch.from_numpy(np.load(DATA_DIR / "AC_data_output.npy"))
        .float()
        .unsqueeze(-1)
    )
    return x_data, y_data


def plot_dataset_sample(input_data: torch.Tensor, output_data: torch.Tensor, idx: int):
    """Plot one input/output pair from the dataset."""
    plt.figure()
    grid = np.linspace(-1, 1, input_data.shape[1])
    plt.plot(grid, input_data[idx, :, 0].cpu().numpy(), label="input: u(t=0)")
    plt.plot(grid, output_data[idx, :, 0].cpu().numpy(), label="output: u(t=1)")
    plt.grid(True, which="both", ls=":")
    plt.legend()
    save_fig("dataset_sample.png")


class SpectralConv1d(nn.Module):
    """
    1D Fourier layer. It does FFT, linear transform, and Inverse FFT.
    Taken from:
    https://colab.research.google.com/drive/1DBZW3AYwzQaxUoXjRxNQml7ClUQFJCR9?usp=sharing
    """

    def __init__(self, in_channels, out_channels, modes1):
        """
        :param in_channels: (int) dimensionality of input
        :param out_channels: (int) dimensionality of output
        :param modes1: (int) number of modes to keep (highest frequencies)
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1

        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            self.scale
            * torch.rand(in_channels, out_channels, self.modes1, dtype=torch.cfloat)
        )

    def compl_mul1d(self, input, weights):
        """
        1D complex multiplication
        :param input: (torch.Tensor, shape=[batch, in_channel, x]) Input in Fourier space
        :param weights: (torch.Tensor, shape=[in_channel, out_channel, x]) Weights in Fourier space
        :return: (torch.Tensor, shape=[batch, out_channel, x]) Output in Fourier space
        """
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x):
        """
        1D Fourier layer forward function
        :param x: (torch.Tensor, shape=[batchsize, in_channels, number of grid points])
        :return: (torch.Tensor, shape=[batchsize, out_channels, number of grid points])
        """
        batchsize = x.shape[0]
        x_ft = torch.fft.rfft(x, dim=-1)

        out_ft = torch.zeros(
            batchsize,
            self.out_channels,
            x.size(-1) // 2 + 1,
            device=x.device,
            dtype=torch.cfloat,
        )
        out_ft[:, :, : self.modes1] = self.compl_mul1d(
            x_ft[:, :, : self.modes1], self.weights1
        )
        x = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)
        return x


class FNO1d(nn.Module):
    """
    Minimal 1D FNO:
    1. Lift the input to the desired channel dimension.
    2. Apply three spectral convolution blocks.
    3. Project back to the output space.
    """

    def __init__(self, modes, width):
        super().__init__()
        self.modes1 = modes
        self.width = width
        self.padding = 1
        self.linear_p = nn.Linear(2, self.width)

        self.spect1 = SpectralConv1d(self.width, self.width, self.modes1)
        self.spect2 = SpectralConv1d(self.width, self.width, self.modes1)
        self.spect3 = SpectralConv1d(self.width, self.width, self.modes1)
        self.lin0 = nn.Conv1d(self.width, self.width, 1)
        self.lin1 = nn.Conv1d(self.width, self.width, 1)
        self.lin2 = nn.Conv1d(self.width, self.width, 1)

        self.linear_q = nn.Linear(self.width, 32)
        self.output_layer = nn.Linear(32, 1)
        self.activation = torch.nn.Tanh()

    def fourier_layer(self, x, spectral_layer, conv_layer):
        return self.activation(spectral_layer(x) + conv_layer(x))

    def linear_layer(self, x, linear_transformation):
        return self.activation(linear_transformation(x))

    def forward(self, x):
        x = self.linear_p(x)
        x = x.permute(0, 2, 1)

        x = self.fourier_layer(x, self.spect1, self.lin0)
        x = self.fourier_layer(x, self.spect2, self.lin1)
        x = self.fourier_layer(x, self.spect3, self.lin2)

        x = x.permute(0, 2, 1)
        x = self.linear_layer(x, self.linear_q)
        x = self.output_layer(x)
        return x


def relative_l2_error(y_true: torch.Tensor, y_pred: torch.Tensor, p: int = 2) -> float:
    """Compute relative L2 error (%)."""
    err = (
        torch.mean((y_true - y_pred).abs() ** p) / torch.mean(y_true.abs() ** p)
    ) ** (1 / p) * 100
    return err.item()


def plot_prediction(x_grid, y_true, y_pred, title, fname, scatter_size=8):
    x_grid_np = np.asarray(x_grid)
    y_true_np = np.asarray(y_true)
    y_pred_np = np.asarray(y_pred)
    plt.figure()
    plt.grid(True, which="both", ls=":")
    plt.plot(x_grid_np, y_true_np, label="True Solution", c="C0", lw=2)
    plt.scatter(
        x_grid_np, y_pred_np, label="Approximate Solution", s=scatter_size, c="orange"
    )
    plt.plot(x_grid_np, y_pred_np, color="orange", label=title)
    plt.legend()
    save_fig(fname)


def main():
    torch.set_default_dtype(torch.float)
    torch.manual_seed(1234)
    np.random.seed(1234)

    device = choose_device()
    print(f"Using device: {device}")

    x_data, y_data = load_data()
    print(f"Total data shape: {x_data.shape}, {y_data.shape}")

    # Splits
    n_train, n_val, n_test = 128, 32, 256
    input_function_train = x_data[:n_train]
    output_function_train = y_data[:n_train]
    input_function_val = x_data[n_train : n_train + n_val]
    output_function_val = y_data[n_train : n_train + n_val]
    input_function_test = x_data[n_train + n_val : n_train + n_val + n_test]
    output_function_test = y_data[n_train + n_val : n_train + n_val + n_test]

    print(f"Train data shape: {input_function_train.shape}, {output_function_train.shape}")
    print(f"Val data shape: {input_function_val.shape}, {output_function_val.shape}")
    print(f"Test data shape: {input_function_test.shape}, {output_function_test.shape}")

    plot_dataset_sample(input_function_train, output_function_train, idx=47)

    batch_size = 10
    input_function_train = input_function_train.to(device)
    output_function_train = output_function_train.to(device)
    input_function_val = input_function_val.to(device)
    output_function_val = output_function_val.to(device)
    input_function_test = input_function_test.to(device)
    output_function_test = output_function_test.to(device)

    train_ds = DictDataset(
        {"x": input_function_train, "y": output_function_train}, name="train"
    )
    dev_ds = DictDataset({"x": input_function_val, "y": output_function_val}, name="dev")
    test_ds = DictDataset(
        {"x": input_function_test, "y": output_function_test}, name="test"
    )

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=train_ds.collate_fn
    )
    dev_loader = torch.utils.data.DataLoader(
        dev_ds, batch_size=batch_size, shuffle=False, collate_fn=dev_ds.collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=test_ds.collate_fn
    )

    modes = 16
    width = 64
    fno_model = FNO1d(modes, width).to(device)

    fno_node = Node(fno_model, ["x"], ["y_fno"], name="fno_model")
    print("symbolic inputs  of the fno_node:", fno_node.input_keys)
    print("symbolic outputs of the fno_node:", fno_node.output_keys)
    print(f"Forward pass shape check: {fno_node(train_ds.datadict)['y_fno'].shape}")

    y_true = variable("y")
    y_hat_fno = variable("y_fno")
    val_constraint = (y_hat_fno == y_true) ^ 2
    val_constraint.update_name("l2_value")

    loss_fno = PenaltyLoss([val_constraint], [])
    problem_fno = Problem(nodes=[fno_node], loss=loss_fno)

    loss_history_callback = LossHistoryCallback(
        plots_dir=PLOTS_DIR / "training", show=False
    )

    learning_rate = 0.001
    epochs = 50
    epoch_verbose = 10
    warmup = 10
    patience = 0

    optimizer = torch.optim.Adam(
        problem_fno.parameters(), lr=learning_rate, weight_decay=1e-5
    )

    trainer = Trainer(
        problem_fno.to(device),
        train_data=train_loader,
        dev_data=dev_loader,
        test_data=test_loader,
        optimizer=optimizer,
        logger=None,
        callback=loss_history_callback,
        epochs=epochs,
        patience=patience,
        epoch_verbose=epoch_verbose,
        train_metric="train_loss",
        dev_metric="dev_loss",
        test_metric="test_loss",
        eval_metric="dev_loss",
        warmup=warmup,
        device=device,
    )

    start_time = time.time()
    best_model = trainer.train()
    print(f"Training wall time: {time.time() - start_time:.2f} seconds")
    problem_fno.load_state_dict(best_model)

    best_outputs = trainer.test(best_model)
    mean_test_loss = best_outputs["mean_test_loss"].detach().cpu().item()
    print(f"Mean test loss: {mean_test_loss}")

    train_loss_history = [l.detach().cpu().item() for l in trainer.loss_history["train"]]
    dev_loss_history = [l.detach().cpu().item() for l in trainer.loss_history["dev"]]

    plt.semilogy(train_loss_history, label="Train loss")
    plt.semilogy(dev_loss_history, label="Dev loss")
    plt.scatter(
        len(train_loss_history),
        mean_test_loss,
        label="Mean test loss",
        c="red",
        marker="x",
    )
    plt.xlabel("# Epochs")
    plt.legend()
    save_fig("training_history_fno.png")

    # Evaluate trained FNO on a test sample
    idx_data = 59
    fno_eval = problem_fno.nodes[0].to(device)
    sample = test_ds[idx_data]
    x_n = sample["x"].unsqueeze(0).to(device)
    y_true_sample = sample["y"].unsqueeze(0).to(device)

    with torch.no_grad():
        y_pred_sample = fno_eval({"x": x_n})["y_fno"]

    x_grid = x_n[0, :, 1].detach().cpu()
    y_true_plot = y_true_sample[0, :, 0].detach().cpu()
    y_pred_plot = y_pred_sample[0, :, 0].detach().cpu()

    err = relative_l2_error(y_true_plot, y_pred_plot)
    print(f"Relative L2 error (full resolution): {err}")
    plot_prediction(
        x_grid,
        y_true_plot,
        y_pred_plot,
        title=f"resolution = {y_true_sample.shape[1]}",
        fname="prediction_full_resolution.png",
        scatter_size=8,
    )

    # Resolution invariance test: subsample grid
    subsample = 10
    x_n_sub = sample["x"][::subsample].unsqueeze(0).to(device)
    y_true_sub = sample["y"][::subsample].unsqueeze(0).to(device)
    res_sub = y_true_sub.shape[1]

    with torch.no_grad():
        y_pred_sub = fno_eval({"x": x_n_sub})["y_fno"]

    x_grid_sub = x_n_sub[0, :, 1].detach().cpu()
    y_true_plot_sub = y_true_sub[0, :, 0].detach().cpu()
    y_pred_plot_sub = y_pred_sub[0, :, 0].detach().cpu()

    err_sub = relative_l2_error(y_true_plot_sub, y_pred_plot_sub)
    print(f"Relative L2 error (subsampled resolution): {err_sub}")
    plot_prediction(
        x_grid_sub,
        y_true_plot_sub,
        y_pred_plot_sub,
        title=f"resolution = {res_sub}",
        fname="prediction_subsampled.png",
        scatter_size=15,
    )


if __name__ == "__main__":
    main()
