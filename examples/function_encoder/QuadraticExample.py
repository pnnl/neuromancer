import torch
import matplotlib.pyplot as plt

from neuromancer.dataset import DictDataset
from neuromancer.system import Node
from neuromancer.constraint import variable
from neuromancer.loss import PenaltyLoss
from neuromancer.problem import Problem
from neuromancer.trainer import Trainer


from src.neuromancer.modules.FunctionEncoder import FunctionEncoder


def create_dataset(f, device='cpu',test=False):

    # batch size
    n_functions = 100
    n_datapoints = 1000

    # samples from -1 to 1
    example_xs = torch.rand(n_functions, n_datapoints, 1) * 2 - 1
    if not test:
        query_xs = torch.rand(n_functions, n_datapoints, 1) * 2 - 1
    else:
        query_xs = torch.linspace(-1, 1, n_datapoints).reshape(-1, 1).repeat(n_functions, 1, 1)

    # sample a,b,c from -3 to 3 for n_functions size
    a = torch.rand(n_functions, 1, 1) * 6 - 3
    b = torch.rand(n_functions, 1, 1) * 6 - 3
    c = torch.rand(n_functions, 1, 1) * 6 - 3

    example_ys = f(example_xs, a, b, c)
    query_ys = f(query_xs, a, b, c)


    dataset = {"example_xs": example_xs.to(device),
                "example_ys": example_ys.to(device),
                "query_xs": query_xs.to(device),
                "query_ys": query_ys.to(device),
                "a": a.to(device),  # NOTE: a,b,c are only used for plotting.
                "b": b.to(device),
                "c": c.to(device),
                }

    return dataset


# Function to sample from
f = lambda x, a, b, c: a * x ** 2 + b * x + c

# Create dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'
train_dataset = create_dataset(f, device=device)
test_dataset = create_dataset(f, device=device, test=True)

# Wrap data into Neuromancer DictDatasets
train_data = DictDataset(train_dataset, name='train')
dev_data = DictDataset(test_dataset, name='dev')

# Create torch dataloaders with DictDatasets
train_loader = torch.utils.data.DataLoader(train_data, batch_size=20,
                                           collate_fn=train_data.collate_fn,
                                           shuffle=False)
dev_loader = torch.utils.data.DataLoader(dev_data, batch_size=20,
                                         collate_fn=dev_data.collate_fn,
                                         shuffle=False)

# Initialize basis functions for the function encoder.
n_basis = 5
basis_functions = [torch.nn.Sequential(
    torch.nn.Linear(1, 64),
    torch.nn.ReLU(),
    torch.nn.Linear(64, 1),
) for i in range(n_basis)]
# the function encoder class provides convenient methods to calibrate estimates using least squares
# specifically, it uses example data to compute the coefficients of the basis functions,
# then uses a linear combination of the basis functions to estimate the function
function_encoder = FunctionEncoder(basis_functions)


# Symbolic wrapper of the neural nets
function_encoder = Node(function_encoder, ['example_xs', 'example_ys', 'query_xs'], ['query_y_hats', 'gram'], name='function_encoder')

# Define symbolic variables in Neuromancer
# these are the variables used in the loss functions
query_y_hats = variable('query_y_hats')
query_ys = variable('query_ys')
gram = variable('gram')

# Define the losses
# The first ensures the function estimates align with the true function
loss_data = (query_y_hats == query_ys)^2
loss_data.name = "loss_data"

# this prevents the magnitude of the basis functions from growing
loss_gram = (torch.diagonal(gram, dim1=1, dim2=2) == torch.ones_like(torch.diagonal(gram, dim1=1, dim2=2)))^2
loss_gram.name = "loss_gram"

loss = PenaltyLoss(objectives=[loss_data, loss_gram], constraints=[])


# Construct the optimization problems
problem = Problem(nodes=[function_encoder], loss=loss)


# Create trainer
num_epochs = 1000
trainer = Trainer(
    problem.to(device),
    train_data=train_loader,
    dev_data=dev_loader,
    optimizer= torch.optim.Adam(problem.parameters(), lr=1e-3),
    epoch_verbose=50,
    epochs=num_epochs,
    warmup=num_epochs,
    device=device
)


# Train function encoder
best_model = trainer.train()

# get best model
problem.load_state_dict(best_model)
trained_model = problem.nodes[0]

# estimate outputs and plot.
with torch.no_grad():
    y_test_pred = trained_model(dev_data.datadict)['query_y_hats'].cpu().detach().numpy()
    query_ys = dev_data.datadict['query_ys'].cpu().detach().numpy()

    fig, axs = plt.subplots(3, 3, figsize=(15, 10))
    for i in range(9):
        # plot the estimates
        ax = axs[i // 3, i % 3]
        ax.plot(dev_data.datadict['query_xs'][i].cpu().detach().numpy(), y_test_pred[i], label='Predicted')
        ax.plot(dev_data.datadict['query_xs'][i].cpu().detach().numpy(), query_ys[i], label='True')

        a = dev_data.datadict['a'][i].cpu().detach().item()
        b = dev_data.datadict['b'][i].cpu().detach().item()
        c = dev_data.datadict['c'][i].cpu().detach().item()
        ax.set_title(f"$y = {a:0.1f}x^2 + {b:0.1f}x + {c:0.1f}$")

        # set labels
        if i // 3 == 2:
            ax.set_xlabel('x')
        if i % 3 == 0:
            ax.set_ylabel('y')

        # set legend
        if i == 0:
            ax.legend(loc='upper right')

    # save or show
    fig.savefig("test.png")
