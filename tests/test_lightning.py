import pytest
import torch
import lightning.pytorch as pl
from neuromancer.problem import LitProblem
from unittest.mock import MagicMock


# Custom hooks definitions
def custom_train_epoch_end(self): 
    if not hasattr(self, 'train_loss_epoch_history') or self.train_loss_epoch_history is None:
        self.train_loss_epoch_history = []
    epoch_average = torch.stack(self.training_step_outputs).mean()
    self.train_loss_epoch_history.append(epoch_average)

def custom_validation_epoch_end(self):
    if not hasattr(self, 'val_loss_epoch_history') or self.val_loss_epoch_history is None:
        self.val_loss_epoch_history = []
    epoch_average = torch.stack(self.validation_step_outputs).mean()
    self.val_loss_epoch_history.append(epoch_average)

def on_train_start(self): 
    print("HELLO WORLD")

def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    return [optimizer], [scheduler]

custom_hooks = {
    'on_train_epoch_end': custom_train_epoch_end,
    'on_validation_epoch_end': custom_validation_epoch_end, 
    'on_train_start': on_train_start, 
    'configure_optimizers': configure_optimizers
}

@pytest.fixture
def problem():
    return MagicMock()

@pytest.fixture
def batch():
    return MagicMock()

@pytest.fixture
def custom_hooks_fixture():
    return custom_hooks

def test_initialization(problem):
    lit_problem = LitProblem(problem)
    assert lit_problem.problem == problem
    assert lit_problem.train_metric == 'train_loss'
    assert lit_problem.lr == 0.001
    assert lit_problem.custom_optimizer is None

def test_training_step(problem, batch):
    lit_problem = LitProblem(problem)
    output = {'train_loss': torch.tensor(1.0)}
    problem.return_value = output

    loss = lit_problem.training_step(batch, 0)
    assert loss == output['train_loss']
    assert loss in lit_problem.training_step_outputs


def test_on_train_epoch_end(problem, custom_hooks_fixture):
    lit_problem = LitProblem(problem, custom_hooks=custom_hooks_fixture)
    lit_problem.training_step_outputs = [torch.tensor(1.0), torch.tensor(2.0)]

    lit_problem.on_train_epoch_end()
    assert torch.tensor(1.5) in lit_problem.train_loss_epoch_history

def test_validate_hooks_signature_error(problem):
    # Define a custom hook with an incorrect signature
    def incorrect_signature_hook(x):  # Should be 'self'
        pass

    def incorrect_signature_hook_2(self, batch, idx):  # Should be 'self, batch, batch_idx'
        pass
    
    # Create a hooks dictionary with the incorrect hook
    incorrect_hooks = {
        'on_train_epoch_end': incorrect_signature_hook,
        'on_train_batch_start': incorrect_signature_hook_2
    }
    
    # Check that the ValueError is raised with the correct message
    with pytest.raises(ValueError):
        LitProblem(problem, custom_hooks=incorrect_hooks)


