import torch
import neuromancer.slim as slim

if __name__ == '__main__':
    """
    Tests
    """
    square = torch.rand(8, 8)
    long = torch.rand(3, 8)
    tall = torch.rand(8, 3)

    for linear in set(list(slim.maps.values())) - slim.square_maps:
        print(linear)
        map = linear(3, 5)
        x = map(tall)
        assert (x.shape[0], x.shape[1]) == (8, 5)
        if not linear is slim.TrivialNullSpaceLinear:
            map = linear(8, 3)
            x = map(long)
            assert (x.shape[0], x.shape[1]) == (3, 3)

    for linear in slim.square_maps:
        print(linear)
        map = linear(8, 8)
        x = map(square)
        assert (x.shape[0], x.shape[1]) == (8, 8)
        x = map(long)
        assert (x.shape[0], x.shape[1]) == (3, 8)
