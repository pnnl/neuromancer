from neuromancer.constraint import variable
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


if __name__ == "__main__":
    torch.manual_seed(0)
    t_x_true = torch.arange(0.0, 1.0 + 0.1, 0.1)
    t_w_true = torch.tensor(0.5)
    t_b_true = torch.tensor(1.0)
    t_y_true = t_x_true * t_w_true + t_b_true

    # we need underlying tensor for
    # 1: tensor model
    # 2: variable model
    t_w_1 = torch.randn(1, requires_grad=True)
    t_b_1 = torch.randn(1, requires_grad=True)
    t_w_2 = t_w_1.clone().detach().requires_grad_()
    t_b_2 = t_b_1.clone().detach().requires_grad_()

    # pure PyTorch model
    opt = torch.optim.Adam([t_w_1, t_b_1])  # parameters of tensor model

    losses = []

    for i in range(10000):
        opt.zero_grad()
        t_y = t_x_true * t_w_1 + t_b_1
        loss = F.mse_loss(t_y, t_y_true)
        losses.append(loss.item())
        loss.backward(retain_graph=True)
        opt.step()

    print(f"w estimated to {t_w_1} true value is {t_w_true}")
    print(f"b estimated to {t_b_1} true value is {t_b_true}")
    plt.plot(losses)
    plt.savefig('regression_loss.png')
    plt.close()
    # Neuromancer.Variable model

    var_x = variable("x")
    var_w = variable(t_w_2, display_name="w")
    var_b = variable(t_b_2, display_name="b")
    var_y_est = var_x * var_w + var_b
    var_y_true = variable(t_y_true, display_name="y_true")  # TODO currently ground truth must be wrapped

    var_loss = F.mse_loss(var_y_est, var_y_true)

    var_loss.draw(figname='loss_computation.png')
    opt_var = torch.optim.Adam(var_loss.parameters())  # parameters of variable model
    losses = []

    for i in range(10000):
        opt_var.zero_grad()
        loss = var_loss({'x': t_x_true})
        losses.append(loss.item())
        loss.backward(retain_graph=True)
        opt_var.step()

    print(f"w estimated to {t_w_2} true value is {t_w_true}")
    print(f"b estimated to {t_b_2} true value is {t_b_true}")
    plt.plot(losses)
    plt.savefig('regression_variables_loss.png')
    plt.close()