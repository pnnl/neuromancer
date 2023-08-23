from neuromancer.psl.base import EmulatorBase, ODE_NonAutonomous, ODE_Autonomous
from neuromancer.psl.building_envelope import systems
import dill
from neuromancer.psl.signals import sines
import torch


def system_to_psl(nm_model, psl_system):
    """

    :param nm_model: (neuromancer.System)
    :param psl_system: (neuromancer.psl.BuildingEnvelope)
    :return: (SystemPSL) An instance of class SystemPSL
    """
    def equations(y, u, d):
        # internally handle the normalization of U and D
        x = torch.cat([y.reshape(1, -1), u.reshape(1, -1), d.reshape(1, -1)], dim=-1)
        y = nm_model({'xn': x})['yn']
        print('y', y.shape)
        return y, y

    # overwrite get_x0
    # overwrite stats


    psl_system.change_backend('torch')
    psl_system.equations = equations
    return psl_system


if __name__ == '__main__':
    sys = systems['SimpleSingleZone']()
    model = torch.load('../../../examples/control/nm_models/best_model_blackbox_0.83_r2.pth', pickle_module=dill)
    model = model.nodes[0].nodes[1]
    sys = system_to_psl(model, sys)
    sys.show(figname='stats.png')
    x0, y0 = sys.get_xy()
    data = sys.simulate(nsim=10, x0=y0, U=sys.get_U(11), D=sys.get_D_obs(11))
    print(data)