import torch

if __name__ == '__main__':
    mb = 1
    sl = 100
    nx = 4
    nh = 2

    model1 = RNN(nx, nh, num_layers=60, nonlinearity=F.gelu, stable=True, bias=True)
    model2 = torch.nn.RNN(nx, nh, num_layers=60, nonlinearity='relu', bias=True)
    inputs = torch.randn(sl, mb, nx)**2 + 20
    print(inputs)
    output, hiddens = model1(inputs)
    output2, hiddens2 = model2(inputs)
    print(output[-1])
    print(output2[-1])